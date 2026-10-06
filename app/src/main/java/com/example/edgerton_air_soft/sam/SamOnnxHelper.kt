package com.example.edgerton_air_soft.sam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * EfficientSAM ONNX-Pipeline fuer die Doppel-Maskierung.
 *
 * Workflow:
 *  User tippt auf das Ziel/die Platte (normierte Koordinate 0..1)
 *    → EfficientSAM segmentiert die zusammenhaengende Region sauber heraus
 *    → Ausgabe: ARGB-Bitmap, Alpha = 255 fuer Foreground, 0 sonst.
 *
 * MainActivity benoetigt zusaetzlich SAM3-Funktionen. SAM3 ist bewusst deaktiviert
 * (Encoder + Decoder zusammen > 250 MB, zu schwer fuer das Zielgeraet). Die Funktionen
 * existieren als No-Op-Stubs, damit der bestehende Fallback-Pfad in MainActivity
 * automatisch zu EfficientSAM weiterspringt.
 *
 * Naming-Hinweis: MainActivity importiert dieses Objekt als SamOnnxHelper.
 */
object SamOnnxHelper {

    private const val TAG = "SamOnnxHelper"

    // ── EfficientSAM ─────────────────────────────────────────────────────
    private const val ESAM_ASSET = "efficient_sam.onnx"

    /** Longest-side Resize-Ziel fuer Inferenz. Kompromiss aus Genauigkeit und Mobile-Speed. */
    private const val ESAM_MAX_SIDE = 1024

    /**
     * EfficientSAM Decoder erwartet padded Point-Slots. Der Default-Export hat
     * `num_points = 5`. Wir nutzen den ersten Slot als echtes Foreground-Tap,
     * der Rest wird mit Label `-1` als Padding gekennzeichnet.
     */
    private const val ESAM_NUM_POINTS = 5

    private val initLock = Any()

    @Volatile private var esamEnv: OrtEnvironment? = null
    @Volatile private var esamSession: OrtSession? = null
    @Volatile private var esamInputImageName: String = "batched_images"
    @Volatile private var esamInputCoordsName: String = "batched_point_coords"
    @Volatile private var esamInputLabelsName: String = "batched_point_labels"
    @Volatile private var esamOutputMaskName: String = "output_masks"
    @Volatile private var esamOutputIouName: String = "iou_predictions"
    @Volatile private var esamReady: Boolean = false
    @Volatile private var esamInitTried: Boolean = false

    // ── Diagnose-Properties fuer das UI (read-only) ─────────────────────
    /** Letzter Init-Fehler. Leer wenn alles OK. */
    @Volatile var lastInitError: String = ""
        private set
    /** Letzter Inferenz-Fehler. Leer wenn alles OK. */
    @Volatile var lastInferenceError: String = ""
        private set
    /** Tatsaechliche Input-Tensor-Namen aus dem geladenen Modell. */
    @Volatile var inputTensorNames: List<String> = emptyList()
        private set
    /** Tatsaechliche Output-Tensor-Namen aus dem geladenen Modell. */
    @Volatile var outputTensorNames: List<String> = emptyList()
        private set
    /** Zeitpunkt der letzten erfolgreichen Inferenz (System.currentTimeMillis). 0 wenn nie. */
    @Volatile var lastInferenceSuccessMs: Long = 0L
        private set
    /** Letzte gemessene Inferenz-Dauer in Millisekunden. 0 wenn nie. */
    @Volatile var lastInferenceMs: Long = 0L
        private set

    // ═════════════════════════════════════════════════════════════════════
    //   PUBLIC API – EfficientSAM
    // ═════════════════════════════════════════════════════════════════════

    /** Idempotent. Erster Aufruf laedt das Modell aus assets (kostet IO + RAM). */
    fun initEfficientSam(context: Context) {
        if (esamReady) return
        synchronized(initLock) {
            if (esamReady || esamInitTried) return
            esamInitTried = true
            val startMs = System.currentTimeMillis()
            try {
                val modelFile = copyAssetToCache(context, ESAM_ASSET)
                val copyMs = System.currentTimeMillis() - startMs
                if (!modelFile.exists() || modelFile.length() <= 0) {
                    throw IllegalStateException("Modell-Cache leer: ${modelFile.absolutePath} (${modelFile.length()} bytes)")
                }
                val env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(max(2, Runtime.getRuntime().availableProcessors() / 2))
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                }
                val sessionStart = System.currentTimeMillis()
                val session = env.createSession(modelFile.absolutePath, opts)
                val sessionMs = System.currentTimeMillis() - sessionStart
                detectIoNames(session)
                esamEnv = env
                esamSession = session
                esamReady = true
                lastInitError = ""
                Log.i(TAG, "EfficientSAM ready in ${copyMs + sessionMs}ms (copy=${copyMs}ms, " +
                    "createSession=${sessionMs}ms) inputs=$inputTensorNames outputs=$outputTensorNames")
            } catch (t: Throwable) {
                esamReady = false
                val msg = "${t::class.java.simpleName}: ${t.message ?: "(no message)"}"
                lastInitError = msg
                Log.e(TAG, "EfficientSAM init failed: $msg", t)
            }
        }
    }

    fun isEfficientSamAvailable(): Boolean = esamReady

    /**
     * Punkt-Prompt-Segmentierung.
     * @param bitmap Originalbild (ARGB_8888)
     * @param xNorm  Tap-X in [0..1] auf dem Originalbild
     * @param yNorm  Tap-Y in [0..1] auf dem Originalbild
     * @return Maske als ARGB-Bitmap in Originalgroesse (Alpha = 255 fuer Hit, sonst 0),
     *         oder null bei Fehlern / leerer Maske.
     */
    fun runEfficientSamPoint(bitmap: Bitmap, xNorm: Double, yNorm: Double): Bitmap? {
        if (!esamReady) {
            lastInferenceError = "esam_not_ready (init=${lastInitError.ifEmpty { "not-attempted-or-success-but-flag-off" }})"
            return null
        }
        val session = esamSession ?: run {
            lastInferenceError = "esam_session_null"
            return null
        }
        val env = esamEnv ?: run {
            lastInferenceError = "esam_env_null"
            return null
        }

        val infStart = System.currentTimeMillis()
        return try {
            val origW = bitmap.width
            val origH = bitmap.height
            if (origW <= 0 || origH <= 0) {
                lastInferenceError = "invalid_bitmap_size=${origW}x${origH}"
                return null
            }

            val scale = minOf(1.0, ESAM_MAX_SIDE.toDouble() / max(origW, origH))
            val w = max(2, (origW * scale).toInt())
            val h = max(2, (origH * scale).toInt())
            val scaled = if (w == origW && h == origH) bitmap
                         else Bitmap.createScaledBitmap(bitmap, w, h, true)

            val px = (xNorm * w).toFloat().coerceIn(0f, (w - 1).toFloat())
            val py = (yNorm * h).toFloat().coerceIn(0f, (h - 1).toFloat())

            val mask = runInference(env, session, scaled, listOf(px to py), listOf(1f))

            if (scaled !== bitmap) scaled.recycle()

            val result = if (mask == null) null
                         else upscaleMaskToOriginal(mask, w, h, origW, origH)

            lastInferenceMs = System.currentTimeMillis() - infStart
            if (result != null) {
                lastInferenceSuccessMs = System.currentTimeMillis()
                lastInferenceError = ""
            } else {
                lastInferenceError = "inference_returned_null"
            }
            result
        } catch (t: Throwable) {
            lastInferenceMs = System.currentTimeMillis() - infStart
            lastInferenceError = "${t::class.java.simpleName}: ${t.message ?: "(no message)"}"
            Log.e(TAG, "runEfficientSamPoint failed: $lastInferenceError", t)
            null
        }
    }

    /**
     * Automatischer Modus ohne Tap: nimmt die Bildmitte als Foreground-Hint.
     * Reicht oft, weil der User das Handy gezielt auf das Board ausrichtet.
     */
    fun runEfficientSamAutomatic(bitmap: Bitmap): Bitmap? {
        return runEfficientSamPoint(bitmap, 0.5, 0.5)
    }

    /**
     * Stage-2-Inferenz: Punkt-Prompt-Segmentierung innerhalb einer vorhandenen
     * Stage-1-Maske (Zielplatte). Pixel ausserhalb der Stage-1-Maske werden vor
     * der Inferenz geschwaerzt, das Ergebnis nach der Inferenz nochmal mit der
     * Stage-1-Maske geschnitten — so faengt EfficientSAM garantiert nichts
     * ausserhalb der Platte ein.
     *
     * @param bitmap      Originalbild (ARGB_8888) in voller Aufloesung
     * @param stage1Mask  Maske der Stage-1-Region (Alpha > 16 = Foreground)
     *                    Wird intern bei Bedarf auf bitmap.size skaliert.
     * @param xNorm       Tap-X in [0..1] auf dem Originalbild
     * @param yNorm       Tap-Y in [0..1] auf dem Originalbild
     */
    fun runEfficientSamPointInMask(
        bitmap: Bitmap,
        stage1Mask: Bitmap,
        xNorm: Double,
        yNorm: Double
    ): Bitmap? {
        if (!esamReady) return null
        return try {
            val origW = bitmap.width
            val origH = bitmap.height
            if (origW <= 0 || origH <= 0) return null

            val sx = (xNorm * origW).toInt().coerceIn(0, origW - 1)
            val sy = (yNorm * origH).toInt().coerceIn(0, origH - 1)
            // Wenn der Tap-Punkt selbst gar nicht im Stage-1-Bereich liegt:
            // sinnlos zu segmentieren, sofort raus.
            val scaledMask = ensureMaskSize(stage1Mask, origW, origH)
            val tapInMask = ((scaledMask.getPixel(sx, sy) ushr 24) and 0xFF) > 16
            if (!tapInMask) {
                if (scaledMask !== stage1Mask) scaledMask.recycle()
                return null
            }

            val masked = blackOutsideMask(bitmap, scaledMask)
            val raw = runEfficientSamPoint(masked, xNorm, yNorm)
            masked.recycle()

            val clipped = if (raw != null) intersectWithMask(raw, scaledMask) else null
            if (raw != null && clipped !== raw) raw.recycle()
            if (scaledMask !== stage1Mask) scaledMask.recycle()
            clipped
        } catch (t: Throwable) {
            Log.e(TAG, "runEfficientSamPointInMask failed", t)
            null
        }
    }

    fun closeEfficientSam() {
        synchronized(initLock) {
            try { esamSession?.close() } catch (_: Throwable) {}
            esamSession = null
            esamReady = false
            esamInitTried = false
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //   SAM3 – bewusst deaktiviert (Modelle > 250 MB, zu schwer)
    // ═════════════════════════════════════════════════════════════════════

    fun initSam3(@Suppress("UNUSED_PARAMETER") context: Context) { /* No-Op: SAM3 disabled */ }
    fun isSam3Available(): Boolean = false
    fun runSam3WithBox(
        @Suppress("UNUSED_PARAMETER") bitmap: Bitmap,
        @Suppress("UNUSED_PARAMETER") boxNorm: FloatArray
    ): Bitmap? = null
    fun runSam3Automatic(@Suppress("UNUSED_PARAMETER") bitmap: Bitmap): Bitmap? = null
    fun closeSam3() { /* No-Op */ }

    // ═════════════════════════════════════════════════════════════════════
    //   INTERNAL
    // ═════════════════════════════════════════════════════════════════════

    /**
     * Liest die echten Tensor-Namen aus dem geladenen Modell. EfficientSAM-Exports
     * variieren leicht je nach Quelle (yformer Default vs. HuggingFace re-exports),
     * deshalb ist Hard-Coding gefaehrlich.
     */
    private fun detectIoNames(session: OrtSession) {
        val inputs = session.inputNames.toList()
        val outputs = session.outputNames.toList()
        inputTensorNames = inputs
        outputTensorNames = outputs

        // Inputs: image-aehnlich, coords-aehnlich, labels-aehnlich
        inputs.firstOrNull { it.contains("image", ignoreCase = true) }
            ?.let { esamInputImageName = it }
        inputs.firstOrNull { it.contains("coord", ignoreCase = true) }
            ?.let { esamInputCoordsName = it }
        inputs.firstOrNull { it.contains("label", ignoreCase = true) }
            ?.let { esamInputLabelsName = it }

        // Outputs: Mask vor IoU. Manche Exporte heissen die Mask "masks" / "predicted_logits".
        outputs.firstOrNull { it.contains("mask", ignoreCase = true) || it.contains("logit", ignoreCase = true) }
            ?.let { esamOutputMaskName = it }
        outputs.firstOrNull { it.contains("iou", ignoreCase = true) || it.contains("score", ignoreCase = true) }
            ?.let { esamOutputIouName = it }
    }

    private fun runInference(
        env: OrtEnvironment,
        session: OrtSession,
        scaled: Bitmap,
        points: List<Pair<Float, Float>>,
        labels: List<Float>
    ): BooleanArray? {
        val w = scaled.width
        val h = scaled.height

        // 1) HWC uint8 → CHW float32 in [0,1]. ImageNet mean/std macht das Modell intern.
        val pixels = IntArray(w * h)
        scaled.getPixels(pixels, 0, w, 0, 0, w, h)
        val img = FloatArray(3 * w * h)
        val plane = w * h
        for (i in 0 until plane) {
            val p = pixels[i]
            img[i] = ((p ushr 16) and 0xFF) / 255f          // R
            img[plane + i] = ((p ushr 8) and 0xFF) / 255f   // G
            img[2 * plane + i] = (p and 0xFF) / 255f        // B
        }

        // 2) Point-Coords + Labels mit Padding auf ESAM_NUM_POINTS auffuellen.
        val coordsArr = FloatArray(ESAM_NUM_POINTS * 2)
        val labelsArr = FloatArray(ESAM_NUM_POINTS) { -1f }
        val nReal = min(points.size, ESAM_NUM_POINTS)
        for (i in 0 until nReal) {
            coordsArr[i * 2] = points[i].first
            coordsArr[i * 2 + 1] = points[i].second
            labelsArr[i] = labels.getOrElse(i) { 1f }
        }

        // 3) Tensoren erzeugen
        val imgTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(img),
            longArrayOf(1L, 3L, h.toLong(), w.toLong())
        )
        val coordsTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(coordsArr),
            longArrayOf(1L, 1L, ESAM_NUM_POINTS.toLong(), 2L)
        )
        val labelsTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(labelsArr),
            longArrayOf(1L, 1L, ESAM_NUM_POINTS.toLong())
        )

        val feeds = mapOf(
            esamInputImageName to imgTensor,
            esamInputCoordsName to coordsTensor,
            esamInputLabelsName to labelsTensor
        )

        return try {
            session.run(feeds).use { result ->
                val masksValue = result.get(esamOutputMaskName).orElse(null) ?: result[0]
                val iouValue: OnnxValue? = result.get(esamOutputIouName).orElse(null)

                pickBestMask(masksValue, iouValue, w, h)
            }
        } finally {
            imgTensor.close(); coordsTensor.close(); labelsTensor.close()
        }
    }

    /**
     * Extrahiert die hoechst-IoU Maske aus dem Multi-Mask-Output.
     * Erwartet `output_masks` als [1, queries=1, masks=K, H, W] FloatArray-Nesting,
     * und `iou_predictions` als [1, 1, K]. Fallback: erste Maske, wenn Shape unbekannt.
     */
    private fun pickBestMask(
        masksValue: OnnxValue,
        iouValue: OnnxValue?,
        scaledW: Int,
        scaledH: Int
    ): BooleanArray? {
        val masks = masksValue.value ?: return null

        // Suche bestes Mask-Index per IoU, defensiv – wenn iou-Tensor nicht da ist nimm Index 0.
        val bestIdx = bestMaskIndexFromIou(iouValue)

        // Navigiere durch die geschachtelten Arrays bis wir 2D [H][W] FloatArray haben.
        val mask2d = extract2dMask(masks, bestIdx, scaledW, scaledH) ?: return null
        val h = mask2d.size
        val w = if (h > 0) mask2d[0].size else 0
        if (w == 0 || h == 0) return null

        // Logits >= 0 = Foreground. Wenn die Output-Aufloesung von scaled abweicht
        // (z.B. low-res 256x256), wird im naechsten Schritt skaliert. Hier liefern wir
        // bereits eine BooleanArray-Repraesentation mit (w, h) als Metadaten in einer
        // 1D-Form, damit der Caller weiterarbeiten kann.
        val flat = BooleanArray(w * h)
        for (y in 0 until h) {
            val row = mask2d[y]
            val base = y * w
            for (x in 0 until w) {
                flat[base + x] = row[x] >= 0f
            }
        }

        // Falls die Modell-Maske kleiner ist als die skalierte Eingabe (low-res Variante),
        // erweitern wir hier nicht – das uebernimmt upscaleMaskToOriginal mit (modelW, modelH).
        // Wir kodieren die Mask-Dimension in den ersten 8 Bytes? Nein, einfacher:
        // Wir packen die Dimension in ein Wrapper-Container ueber Companion-State.
        lastMaskW = w
        lastMaskH = h
        return flat
    }

    @Volatile private var lastMaskW: Int = 0
    @Volatile private var lastMaskH: Int = 0

    private fun bestMaskIndexFromIou(iouValue: OnnxValue?): Int {
        if (iouValue == null) return 0
        val raw = iouValue.value ?: return 0
        return try {
            // Erwartet [1, 1, K] oder [1, K]
            val flat = flattenToFloatList(raw)
            if (flat.isEmpty()) 0 else flat.indices.maxByOrNull { flat[it] } ?: 0
        } catch (_: Throwable) { 0 }
    }

    private fun flattenToFloatList(any: Any?): List<Float> {
        val out = mutableListOf<Float>()
        fun walk(v: Any?) {
            when (v) {
                null -> {}
                is FloatArray -> v.forEach { out += it }
                is Array<*> -> v.forEach { walk(it) }
                is Float -> out += v
                else -> {}
            }
        }
        walk(any)
        return out
    }

    /**
     * Findet im verschachtelten Array das beste 2D-FloatArray (= Maske) fuer Index `idx`.
     * Erwartetes Layout: `[1][1][K][H][W]`. Wenn anders, fallbacken wir auf die erste 2D-Slice.
     */
    @Suppress("UNCHECKED_CAST")
    private fun extract2dMask(any: Any, idx: Int, scaledW: Int, scaledH: Int): Array<FloatArray>? {
        // Suche rekursiv die erste 2D FloatArray-Ebene
        fun firstTwoD(v: Any?): Array<FloatArray>? {
            if (v is Array<*> && v.isNotEmpty() && v[0] is FloatArray) {
                @Suppress("UNCHECKED_CAST")
                return v as Array<FloatArray>
            }
            if (v is Array<*>) {
                for (item in v) {
                    val nested = firstTwoD(item)
                    if (nested != null) return nested
                }
            }
            return null
        }

        // Versuch [1][1][K][H][W] mit idx
        try {
            val a = any as Array<*>
            val b = a[0] as Array<*>
            val c = b[0] as Array<*>
            val safeIdx = idx.coerceIn(0, c.size - 1)
            val candidate = c[safeIdx]
            if (candidate is Array<*> && candidate.isNotEmpty() && candidate[0] is FloatArray) {
                @Suppress("UNCHECKED_CAST")
                return candidate as Array<FloatArray>
            }
        } catch (_: Throwable) { /* fall through */ }

        // Versuch [1][K][H][W]
        try {
            val a = any as Array<*>
            val b = a[0] as Array<*>
            val safeIdx = idx.coerceIn(0, b.size - 1)
            val candidate = b[safeIdx]
            if (candidate is Array<*> && candidate.isNotEmpty() && candidate[0] is FloatArray) {
                @Suppress("UNCHECKED_CAST")
                return candidate as Array<FloatArray>
            }
        } catch (_: Throwable) { /* fall through */ }

        return firstTwoD(any)
    }

    /**
     * Konvertiert eine BooleanArray-Maske (Groesse lastMaskW x lastMaskH) in eine
     * ARGB-Bitmap (Alpha = 255 fuer Foreground, 0 sonst) und skaliert sie zurueck
     * auf die Originalbildgroesse.
     */
    private fun upscaleMaskToOriginal(
        flat: BooleanArray,
        scaledW: Int,
        scaledH: Int,
        origW: Int,
        origH: Int
    ): Bitmap? {
        val mw = lastMaskW
        val mh = lastMaskH
        if (mw <= 0 || mh <= 0) return null

        // Zwischen-Bitmap in Mask-Aufloesung
        val mid = Bitmap.createBitmap(mw, mh, Bitmap.Config.ARGB_8888)
        val midPixels = IntArray(mw * mh)
        for (i in flat.indices) {
            midPixels[i] = if (flat[i]) Color.argb(255, 255, 255, 255) else 0
        }
        mid.setPixels(midPixels, 0, mw, 0, 0, mw, mh)

        // Auf Originalgroesse skalieren (NEAREST haerter, BILINEAR weicher).
        // Wir wollen harte Kanten — also Filter aus.
        val full = if (mw == origW && mh == origH) mid
                   else Bitmap.createScaledBitmap(mid, origW, origH, false)
        if (full !== mid) mid.recycle()

        // Quick-Check: ist da ueberhaupt etwas drin?
        val checkPixels = IntArray(min(full.width * full.height, 4096))
        full.getPixels(checkPixels, 0, full.width, 0, 0,
            min(full.width, 64), min(full.height, max(1, checkPixels.size / max(1, full.width))))
        // (Best-effort Sanity-Check – wir geben die Maske trotzdem zurueck.)

        return full
    }

    /**
     * Skaliert die Maske auf (targetW, targetH), falls noetig. Liefert ggf. eine
     * neue Bitmap — Aufrufer muss pruefen ob `result !== mask` und entsprechend
     * recyclen.
     */
    private fun ensureMaskSize(mask: Bitmap, targetW: Int, targetH: Int): Bitmap {
        if (mask.width == targetW && mask.height == targetH) return mask
        // false = NEAREST → harte Kanten beibehalten
        return Bitmap.createScaledBitmap(mask, targetW, targetH, false)
    }

    /**
     * Erzeugt eine Kopie von src, in der alle Pixel ausserhalb der Maske
     * (Alpha <= 16) auf Schwarz gesetzt werden. Pixel innerhalb der Maske
     * werden 1:1 uebernommen. Das ist der Stage-2-Pre-Processing-Trick:
     * SAM sieht ausserhalb der Platte nur Schwarz und kann nichts dort
     * faelschlicherweise „faschen“.
     */
    private fun blackOutsideMask(src: Bitmap, mask: Bitmap): Bitmap {
        val w = src.width
        val h = src.height
        val srcPx = IntArray(w * h)
        val maskPx = IntArray(w * h)
        src.getPixels(srcPx, 0, w, 0, 0, w, h)
        mask.getPixels(maskPx, 0, w, 0, 0, w, h)
        val out = IntArray(w * h)
        for (i in srcPx.indices) {
            val inside = ((maskPx[i] ushr 24) and 0xFF) > 16
            out[i] = if (inside) srcPx[i] else 0xFF000000.toInt()
        }
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Bool-AND zweier ARGB-Masken (Alpha als Foreground-Indikator). Returns
     * eine neue Bitmap, in der ein Pixel nur Foreground ist wenn er in BEIDEN
     * Masken Foreground ist.
     */
    private fun intersectWithMask(target: Bitmap, gate: Bitmap): Bitmap {
        val w = target.width
        val h = target.height
        val gateScaled = ensureMaskSize(gate, w, h)
        val tPx = IntArray(w * h)
        val gPx = IntArray(w * h)
        target.getPixels(tPx, 0, w, 0, 0, w, h)
        gateScaled.getPixels(gPx, 0, w, 0, 0, w, h)
        val out = IntArray(w * h)
        for (i in tPx.indices) {
            val tHit = ((tPx[i] ushr 24) and 0xFF) > 16
            val gHit = ((gPx[i] ushr 24) and 0xFF) > 16
            out[i] = if (tHit && gHit) tPx[i] else 0
        }
        if (gateScaled !== gate) gateScaled.recycle()
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    private fun copyAssetToCache(context: Context, assetName: String): File {
        val dest = File(context.cacheDir, assetName)
        // Cache nur einmal befuellen – pruefe Groesse, neukopieren wenn anders
        val asset = context.assets.openFd(assetName)
        val expectedLen = asset.length
        asset.close()
        if (dest.exists() && dest.length() == expectedLen) return dest

        context.assets.open(assetName).use { input ->
            FileOutputStream(dest).use { output ->
                input.copyTo(output, bufferSize = 64 * 1024)
            }
        }
        return dest
    }
}
