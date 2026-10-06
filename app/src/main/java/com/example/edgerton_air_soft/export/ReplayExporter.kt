package com.example.edgerton_air_soft.export

import com.example.edgerton_air_soft.audio.AudioWindow
import com.example.edgerton_air_soft.camera.FrameData
import com.example.edgerton_air_soft.trigger.TriggerEvent

data class ExportResult(
    val ok: Boolean,
    val outputPath: String,
    val frameCount: Int,
    val format: String,
    val fallbackUsed: Boolean,
    val audioTrackMuxed: Boolean = false,
    val audioInfo: String? = null,
    val error: String? = null
)

interface ReplayExporter {
    fun export(
        trigger: TriggerEvent,
        frames: List<FrameData>,
        audioWindow: AudioWindow? = null,
        onDone: (ExportResult) -> Unit
    )

    fun shutdown()
}

class FallbackReplayExporter(
    private val primary: ReplayExporter,
    private val fallback: ReplayExporter
) : ReplayExporter {
    override fun export(
        trigger: TriggerEvent,
        frames: List<FrameData>,
        audioWindow: AudioWindow?,
        onDone: (ExportResult) -> Unit
    ) {
        primary.export(trigger, frames, audioWindow) { result ->
            if (result.ok) {
                onDone(result)
                return@export
            }

            fallback.export(trigger, frames, audioWindow) { fallbackResult ->
                if (fallbackResult.ok) {
                    onDone(
                        fallbackResult.copy(
                            fallbackUsed = true,
                            error = result.error
                        )
                    )
                } else {
                    onDone(
                        fallbackResult.copy(
                            fallbackUsed = true,
                            error = "MP4: ${result.error}; Fallback: ${fallbackResult.error}"
                        )
                    )
                }
            }
        }
    }

    override fun shutdown() {
        primary.shutdown()
        fallback.shutdown()
    }
}
