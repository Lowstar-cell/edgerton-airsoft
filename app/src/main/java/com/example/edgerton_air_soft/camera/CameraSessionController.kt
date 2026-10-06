package com.example.edgerton_air_soft.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.SystemClock
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Kamera-Controller mit drei Use-Cases:
 * 1) Preview – Live-Bild auf dem Display
 * 2) ImageAnalysis – Echtzeit-Luma-Frames fuer Motion/Schuss-Erkennung (30 fps, hohe Aufl.)
 * 3) ImageCapture – Hochwertige Standbilder fuer Board-Segmentierung (Farbe + Schaerfe)
 *
 * Kein erzwungenes 120 fps mehr – das hat Belichtung/Qualitaet gekillt.
 * Stattdessen: 30 fps default, Autofokus CONTINUOUS_PICTURE, vernuenftige Aufloesung.
 */
class CameraSessionController(
    private val context: Context
) {
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val captureExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null
    private var preview: Preview? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var camera: Camera? = null
    private var lumaRowScratch: ByteArray? = null
    
    /** Ermittelter maximaler FPS-Range des Geräts (für SuperSlowMo Boost). */
    private var maxSupportedFpsRange: Range<Int> = Range(30, 30)

    /** True sobald die Kamera laeuft. */
    @Volatile var isActive: Boolean = false; private set

    /**
     * Frame-Analyse (ImageAnalysis) nur aktivieren wenn Match laeuft.
     * Preview und ImageCapture sind immer aktiv; nur der onFrame-Callback
     * wird blockiert wenn false – kein CPU- oder Akku-Verbrauch im Standby.
     */
    @Volatile var analysisEnabled: Boolean = false

    @SuppressLint("UnsafeOptInUsageError")
    fun start(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        lowMemoryMode: Boolean = false,
        onFrame: (FrameData) -> Unit,
        onError: (String) -> Unit
    ) {
        if (isActive && camera != null && imageCapture != null && imageAnalysis != null) {
            preview?.surfaceProvider = previewView.surfaceProvider
            return
        }
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener(
            {
                try {
                    val provider = cameraProviderFuture.get()
                    cameraProvider = provider

                    // ── Preview ─────────────────────────────────────────────────
                    // Höhere Vorschau-Auflösung für ein schärferes Bild (1080p)
                    val previewRes = if (lowMemoryMode) Size(1280, 720) else Size(1920, 1080)
                    val previewResSelector = ResolutionSelector.Builder()
                        .setResolutionStrategy(ResolutionStrategy(previewRes, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                        .build()
                    val preview = Preview.Builder()
                        .setResolutionSelector(previewResSelector)
                        .build().also {
                            it.surfaceProvider = previewView.surfaceProvider
                        }
                    this.preview = preview

                    // ImageAnalysis für Echtzeit-Luma-Frames (30 fps)
                    val analysisRes = Size(1280, 720)
                    val resolutionSelector = ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(analysisRes, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
                        )
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                        .build()
                    val analysisBuilder = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .setResolutionSelector(resolutionSelector)

                    // FPS-Range auf (15, 30) begrenzen um Akku zu schonen
                    val fpsRange = Range(15, 30)
                    Camera2Interop.Extender(analysisBuilder).apply {
                        setCaptureRequestOption(
                            android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                            fpsRange
                        )
                        // Autofokus: dauerhaft scharf auf Motiv
                        setCaptureRequestOption(
                            android.hardware.camera2.CaptureRequest.CONTROL_AF_MODE,
                            android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                        )
                        // Auto-Weissabgleich
                        setCaptureRequestOption(
                            android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE,
                            android.hardware.camera2.CameraMetadata.CONTROL_AWB_MODE_AUTO
                        )
                    }

                    val analysis = analysisBuilder.build()
                    imageAnalysis = analysis
                    analysis.setAnalyzer(analysisExecutor) { image ->
                        // Nicht verarbeiten wenn kein Match aktiv – spart CPU/Akku
                        if (!analysisEnabled) { image.close(); return@setAnalyzer }
                        try {
                            val width = image.width
                            val height = image.height
                            val luma = copyLuma(
                                buffer = image.planes[0].buffer,
                                width = width,
                                height = height,
                                rowStride = image.planes[0].rowStride
                            )
                            onFrame(
                                FrameData(
                                    timestampMs = SystemClock.elapsedRealtime(),
                                    width = width,
                                    height = height,
                                    luma = luma
                                )
                            )
                        } catch (t: Throwable) {
                            onError("Frame-Analyse Fehler: ${t.message ?: t::class.java.simpleName}")
                        } finally {
                            image.close()
                        }
                    }

                    // ── ImageCapture ────────────────────────────────────────────
                    // Fuer hochwertige Standbilder (Board-Segmentierung).
                    val capture = ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .build()
                    imageCapture = capture

                    // ── Binden ──────────────────────────────────────────────────
                    provider.unbindAll()
                    camera = provider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis,
                        capture
                    )
                    
                    // Max FPS Range für dieses Gerät ermitteln
                    val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager
                    val backCameraInfo = provider.availableCameraInfos.first { 
                        it.lensFacing == CameraSelector.LENS_FACING_BACK 
                    }
                    @SuppressLint("UnsafeOptInUsageError")
                    val cameraId = androidx.camera.camera2.interop.Camera2CameraInfo.from(backCameraInfo).getCameraId()
                    val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                    val ranges = characteristics.get(android.hardware.camera2.CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                    maxSupportedFpsRange = ranges?.maxByOrNull { it.upper } ?: Range(30, 30)
                    android.util.Log.i("CameraSession", "Max supported FPS range: $maxSupportedFpsRange")

                    isActive = true

                    // Initiales Autofokus-Antippen: Bildmitte fokussieren
                    triggerAutoFocus()

                } catch (t: Throwable) {
                    isActive = false
                    onError("Kamera-Start fehlgeschlagen: ${t.message ?: t::class.java.simpleName}")
                }
            },
            ContextCompat.getMainExecutor(context)
        )
    }

    /**
     * Hochwertige Einzelaufnahme fuer Board-Segmentierung.
     * Liefert ein FrameData mit Luma UND RGBA (Farbbild).
     * Nutzt ImageCapture → scharfes, gut belichtetes Bild.
     */
    fun captureStillFrame(callback: (FrameData?) -> Unit) {
        val capture = imageCapture
        if (capture == null) {
            callback(null)
            return
        }
        capture.takePicture(captureExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    val bitmap = imageProxyToBitmap(image)
                    if (bitmap == null) {
                        callback(null)
                        return
                    }
                    // Bitmap → Luma + RGBA
                    val w = bitmap.width
                    val h = bitmap.height
                    val pixels = IntArray(w * h)
                    bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

                    // Luma aus RGB
                    val luma = ByteArray(w * h)
                    val rgba = ByteArray(w * h * 4)
                    for (i in pixels.indices) {
                        val px = pixels[i]
                        val r = (px shr 16) and 0xFF
                        val g = (px shr 8) and 0xFF
                        val b = px and 0xFF
                        val a = (px shr 24) and 0xFF
                        // BT.601 Luma
                        luma[i] = ((0.299 * r + 0.587 * g + 0.114 * b).toInt().coerceIn(0, 255)).toByte()
                        val j = i * 4
                        rgba[j] = r.toByte()
                        rgba[j + 1] = g.toByte()
                        rgba[j + 2] = b.toByte()
                        rgba[j + 3] = a.toByte()
                    }

                    bitmap.recycle()
                    callback(
                        FrameData(
                            timestampMs = SystemClock.elapsedRealtime(),
                            width = w,
                            height = h,
                            luma = luma,
                            rgba = rgba
                        )
                    )
                } catch (t: Throwable) {
                    callback(null)
                } finally {
                    image.close()
                }
            }

            override fun onError(exception: ImageCaptureException) {
                callback(null)
            }
        })
    }

    /** Fokus auf Bildmitte antippen. */
    fun triggerAutoFocus() {
        val cam = camera ?: return
        try {
            val factory = SurfaceOrientedMeteringPointFactory(1f, 1f)
            val center = factory.createPoint(0.5f, 0.5f)
            val action = FocusMeteringAction.Builder(center, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(3, java.util.concurrent.TimeUnit.SECONDS)
                .build()
            cam.cameraControl.startFocusAndMetering(action)
        } catch (_: Throwable) { /* best-effort */ }
    }

    /**
     * Schaltet kurzzeitig auf hohe FPS um (z.B. für Replay-Aufnahme nach Schuss).
     * Nutzt den maximalen FPS-Range des Geräts (SuperSlowMo).
     */
    fun boostFps(boost: Boolean) {
        val cam = camera ?: return
        if (boost && maxSupportedFpsRange.upper <= 30) return
        // Im Boost-Fall nehmen wir das Maximum was das Gerät kann (z.B. 60, 120 oder 240 fps)
        val fpsRange = if (boost) maxSupportedFpsRange else Range(15, 30)
        
        @SuppressLint("UnsafeOptInUsageError")
        androidx.camera.camera2.interop.Camera2CameraControl.from(cam.cameraControl).setCaptureRequestOptions(
            androidx.camera.camera2.interop.CaptureRequestOptions.Builder()
                .setCaptureRequestOption(android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange)
                .build()
        )
    }

    fun stop() {
        isActive = false
        cameraProvider?.unbindAll()
        camera = null
        preview = null
        imageCapture = null
        imageAnalysis = null
    }

    fun release() {
        stop()
        analysisExecutor.shutdownNow()
        captureExecutor.shutdownNow()
        lumaRowScratch = null
        imageCapture = null
    }

    // ── Hilfsfunktionen ─────────────────────────────────────────────────

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return when (image.format) {
            ImageFormat.JPEG -> {
                val buffer = image.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                val raw = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
                // Rotation anwenden
                applyRotation(raw, image.imageInfo.rotationDegrees)
            }
            ImageFormat.YUV_420_888 -> {
                yuvToRgbBitmap(image)
            }
            else -> null
        }
    }

    private fun applyRotation(bmp: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bmp
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
        if (rotated !== bmp) bmp.recycle()
        return rotated
    }

    private fun yuvToRgbBitmap(image: ImageProxy): Bitmap? {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val ySize = yBuf.remaining()
        val uSize = uBuf.remaining()
        val vSize = vBuf.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuf.get(nv21, 0, ySize)
        // NV21: VU interleaved
        vBuf.get(nv21, ySize, vSize)
        uBuf.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val jpegBytes = out.toByteArray()
        val raw = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null
        return applyRotation(raw, image.imageInfo.rotationDegrees)
    }

    private fun copyLuma(buffer: ByteBuffer, width: Int, height: Int, rowStride: Int): ByteArray {
        val out = ByteArray(width * height)
        buffer.rewind()
        if (rowStride == width) {
            val toCopy = minOf(buffer.remaining(), out.size)
            buffer.get(out, 0, toCopy)
            return out
        }

        val row = lumaRowScratch?.takeIf { it.size >= rowStride } ?: ByteArray(rowStride).also {
            lumaRowScratch = it
        }

        var srcPos = 0
        var dstPos = 0
        repeat(height) {
            if (srcPos + rowStride > buffer.capacity()) return@repeat
            buffer.position(srcPos)
            val readable = minOf(rowStride, buffer.remaining())
            buffer.get(row, 0, readable)
            System.arraycopy(row, 0, out, dstPos, minOf(width, readable))
            srcPos += rowStride
            dstPos += width
        }
        return out
    }
}

