package com.example.edgerton_air_soft.export

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import com.example.edgerton_air_soft.audio.AudioWindow
import com.example.edgerton_air_soft.camera.FrameData
import com.example.edgerton_air_soft.trigger.TriggerEvent
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ImageSequenceExporter(
    private val context: Context
) : ReplayExporter {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()

    override fun export(
        trigger: TriggerEvent,
        frames: List<FrameData>,
        audioWindow: AudioWindow?,
        onDone: (ExportResult) -> Unit
    ) {
        executor.execute {
            try {
                if (frames.isEmpty()) {
                    onDone(
                        ExportResult(
                            ok = false,
                            outputPath = "",
                            frameCount = 0,
                            format = "png-sequence",
                            fallbackUsed = false,
                            error = "Keine Frames im Triggerfenster"
                        )
                    )
                    return@execute
                }

                val root = File(
                    context.getExternalFilesDir(Environment.DIRECTORY_MOVIES),
                    "airsoft_replays"
                )
                if (!root.exists()) root.mkdirs()

                val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(trigger.timestampMs))
                val outDir = File(root, "trigger_${trigger.id}_$stamp")
                outDir.mkdirs()

                frames.forEachIndexed { index, frame ->
                    val bmp = grayscaleBitmap(frame)
                    val file = File(outDir, String.format(Locale.US, "f_%04d.png", index))
                    FileOutputStream(file).use { fos ->
                        bmp.compress(Bitmap.CompressFormat.PNG, 100, fos)
                    }
                    bmp.recycle()
                }

                val meta = File(outDir, "meta.txt")
                meta.writeText(
                    buildString {
                        appendLine("triggerId=${trigger.id}")
                        appendLine("timestampMs=${trigger.timestampMs}")
                        appendLine("reason=${trigger.reason}")
                        appendLine("motionScore=${trigger.motionScore}")
                        appendLine("audioPeakToNoise=${trigger.audioPeakToNoise}")
                        appendLine("frames=${frames.size}")
                        appendLine("audioSamples=${audioWindow?.samples?.size ?: 0}")
                        appendLine("audioSampleRate=${audioWindow?.sampleRate ?: 0}")
                    }
                )

                onDone(
                    ExportResult(
                        ok = true,
                        outputPath = outDir.absolutePath,
                        frameCount = frames.size,
                        format = "png-sequence",
                        fallbackUsed = false,
                        audioTrackMuxed = false,
                        audioInfo = "Kein Audio-Track im PNG-Fallback"
                    )
                )
            } catch (t: Throwable) {
                onDone(
                    ExportResult(
                        ok = false,
                        outputPath = "",
                        frameCount = 0,
                        format = "png-sequence",
                        fallbackUsed = false,
                        audioTrackMuxed = false,
                        audioInfo = "Kein Audio-Track im PNG-Fallback",
                        error = t.message ?: t::class.java.simpleName
                    )
                )
            }
        }
    }

    override fun shutdown() {
        executor.shutdownNow()
    }

    private fun grayscaleBitmap(frame: FrameData): Bitmap {
        val width = frame.width
        val height = frame.height
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val y = frame.luma[i].toInt() and 0xFF
            pixels[i] = -0x1000000 or (y shl 16) or (y shl 8) or y
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }
}
