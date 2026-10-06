package com.example.edgerton_air_soft.export

import android.content.Context
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Environment
import com.example.edgerton_air_soft.audio.AudioWindow
import com.example.edgerton_air_soft.camera.FrameData
import com.example.edgerton_air_soft.trigger.TriggerEvent
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max

class Mp4SlowMotionExporter(
    private val context: Context,
    private val slowMoFactor: Double = 4.0,
    private val targetFps: Int = 30,
    private val bitRate: Int = 2_000_000
) : ReplayExporter {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()

    override fun export(
        trigger: TriggerEvent,
        frames: List<FrameData>,
        audioWindow: AudioWindow?,
        onDone: (ExportResult) -> Unit
    ) {
        executor.execute {
            if (frames.isEmpty()) {
                onDone(
                    ExportResult(
                        ok = false,
                        outputPath = "",
                        frameCount = 0,
                        format = "mp4",
                        fallbackUsed = false,
                        error = "Keine Frames im Triggerfenster"
                    )
                )
                return@execute
            }

            val rootBase = context.getExternalFilesDir(Environment.DIRECTORY_MOVIES)
            if (rootBase == null) {
                onDone(
                    ExportResult(
                        ok = false,
                        outputPath = "",
                        frameCount = 0,
                        format = "mp4",
                        fallbackUsed = false,
                        error = "Exportpfad nicht verfuegbar"
                    )
                )
                return@execute
            }

            val root = File(rootBase, "airsoft_replays")
            if (!root.exists() && !root.mkdirs()) {
                onDone(
                    ExportResult(
                        ok = false,
                        outputPath = "",
                        frameCount = 0,
                        format = "mp4",
                        fallbackUsed = false,
                        error = "Exportordner konnte nicht erstellt werden"
                    )
                )
                return@execute
            }

            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(trigger.timestampMs))
            val outFile = File(root, "trigger_${trigger.id}_$stamp.mp4")

            try {
                val report = encodeToMp4(frames, audioWindow, outFile)
                onDone(
                    ExportResult(
                        ok = true,
                        outputPath = outFile.absolutePath,
                        frameCount = frames.size,
                        format = report.format,
                        fallbackUsed = false,
                        audioTrackMuxed = report.audioTrackMuxed,
                        audioInfo = report.audioInfo
                    )
                )
            } catch (t: Throwable) {
                if (outFile.exists()) outFile.delete()
                onDone(
                    ExportResult(
                        ok = false,
                        outputPath = "",
                        frameCount = 0,
                        format = "mp4",
                        fallbackUsed = false,
                        error = t.message ?: t::class.java.simpleName
                    )
                )
            }
        }
    }

    override fun shutdown() {
        executor.shutdownNow()
    }

    @Suppress("DEPRECATION")
    private fun encodeToMp4(frames: List<FrameData>, audioWindow: AudioWindow?, outFile: File): Mp4EncodingReport {
        val first = frames.first()
        val width = first.width
        val height = first.height

        require(width > 0 && height > 0) { "Ungueltige Videogroesse ${width}x$height" }
        require(width % 2 == 0 && height % 2 == 0) { "Nur gerade Aufloesungen unterstuetzt (${width}x$height)" }
        validateFrames(frames, width, height)

        val audioInputSamples = audioWindow?.samples?.size ?: 0
        var audioInfo = when {
            audioWindow == null -> "Kein Audiofenster vorhanden"
            audioInputSamples == 0 -> "Audiofenster leer"
            else -> null
        }

        val encodedAudio = if (audioWindow != null && audioInputSamples > 0) {
            try {
                val encoded = encodeAudioAac(audioWindow)
                if (encoded.samples.isNotEmpty()) {
                    audioInfo = "AAC kodiert (${encoded.samples.size} Pakete)"
                } else {
                    audioInfo = "AAC kodiert, aber ohne Pakete"
                }
                encoded
            } catch (t: Throwable) {
                audioInfo = "AAC deaktiviert: ${t.message ?: t::class.java.simpleName}"
                null
            }
        } else {
            null
        }

        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        val chosenColorFormat = selectColorFormat(codec.codecInfo)
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, chosenColorFormat)
            setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
            setInteger(MediaFormat.KEY_FRAME_RATE, targetFps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }

        var muxer: MediaMuxer? = null
        var videoTrackIndex = -1
        var audioTrackIndex = -1
        var muxerStarted = false
        var audioWritten = false
        val bufferInfo = MediaCodec.BufferInfo()

        try {
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()
            muxer = MediaMuxer(outFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

            val baseMs = first.timestampMs
            val frameDurationUs = max(1L, (1_000_000.0 / targetFps.toDouble() * slowMoFactor).toLong())
            var inputIndex = 0
            var signaledEos = false
            var lastPtsUs = -1L

            while (true) {
                if (!signaledEos && inputIndex < frames.size) {
                    val inId = codec.dequeueInputBuffer(10_000)
                    if (inId >= 0) {
                        val inBuffer = codec.getInputBuffer(inId) ?: throw IllegalStateException("Input buffer null")
                        inBuffer.clear()

                        val frame = frames[inputIndex]
                        val yuv = lumaToYuv420(frame, chosenColorFormat)
                        if (yuv.size > inBuffer.remaining()) {
                            throw IllegalStateException("Input buffer zu klein: ${inBuffer.remaining()} < ${yuv.size}")
                        }
                        inBuffer.put(yuv)

                        val rawPts = presentationTimeUs(frame.timestampMs - baseMs)
                        val ptsUs = if (rawPts > lastPtsUs) rawPts else lastPtsUs + 1
                        lastPtsUs = ptsUs

                        codec.queueInputBuffer(inId, 0, yuv.size, ptsUs, 0)
                        inputIndex++
                    }
                } else if (!signaledEos) {
                    val inId = codec.dequeueInputBuffer(10_000)
                    if (inId >= 0) {
                        val eosPts = max(lastPtsUs + 1, lastPtsUs + frameDurationUs)
                        codec.queueInputBuffer(inId, 0, 0, eosPts, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        signaledEos = true
                    }
                }

                when (val outId = codec.dequeueOutputBuffer(bufferInfo, 10_000)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit

                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        if (muxerStarted) throw IllegalStateException("Output format changed twice")
                        videoTrackIndex = muxer.addTrack(codec.outputFormat)
                        if (encodedAudio != null && encodedAudio.samples.isNotEmpty()) {
                            audioTrackIndex = muxer.addTrack(encodedAudio.format)
                        }
                        muxer.start()
                        muxerStarted = true

                        if (encodedAudio != null && encodedAudio.samples.isNotEmpty() && !audioWritten) {
                            writeAudioSamples(muxer, audioTrackIndex, encodedAudio.samples)
                            audioWritten = true
                            audioInfo = "AAC gemuxt (${encodedAudio.samples.size} Pakete)"
                        }
                    }

                    else -> {
                        if (outId >= 0) {
                            val outBuffer: ByteBuffer = codec.getOutputBuffer(outId)
                                ?: throw IllegalStateException("Output buffer null")

                            if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                                bufferInfo.size = 0
                            }

                            if (bufferInfo.size > 0) {
                                if (!muxerStarted) throw IllegalStateException("Muxer noch nicht gestartet")
                                outBuffer.position(bufferInfo.offset)
                                outBuffer.limit(bufferInfo.offset + bufferInfo.size)
                                muxer.writeSampleData(videoTrackIndex, outBuffer, bufferInfo)
                            }

                            codec.releaseOutputBuffer(outId, false)
                            if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                        }
                    }
                }
            }
        } finally {
            try {
                codec.stop()
            } catch (_: Throwable) {
            }
            codec.release()

            if (muxerStarted) {
                try {
                    muxer?.stop()
                } catch (_: Throwable) {
                }
            }
            muxer?.release()
        }

        if (encodedAudio != null && encodedAudio.samples.isNotEmpty() && !audioWritten && audioInfo == null) {
            audioInfo = "AAC vorhanden, aber nicht gemuxt"
        }

        return Mp4EncodingReport(
            format = if (audioWritten) "mp4+aac" else "mp4",
            audioTrackMuxed = audioWritten,
            audioInfo = audioInfo
        )
    }

    @Suppress("DEPRECATION")
    private fun encodeAudioAac(window: AudioWindow): EncodedAudio {
        val mime = MediaFormat.MIMETYPE_AUDIO_AAC
        val sampleRate = window.sampleRate
        require(sampleRate > 0) { "Audio sampleRate ungueltig" }

        val format = MediaFormat.createAudioFormat(mime, sampleRate, 1).apply {
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_BIT_RATE, 64_000)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 16_384)
        }

        val codec = MediaCodec.createEncoderByType(mime)
        val info = MediaCodec.BufferInfo()
        val samples = ArrayList<EncodedSample>()

        var outFormat: MediaFormat? = null
        try {
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()

            val blockSamples = 1024
            var sampleOffset = 0
            var signaledEos = false
            var lastPtsUs = -1L

            while (true) {
                if (!signaledEos && sampleOffset < window.samples.size) {
                    val inId = codec.dequeueInputBuffer(10_000)
                    if (inId >= 0) {
                        val inBuffer = codec.getInputBuffer(inId)
                            ?: throw IllegalStateException("AAC input buffer null")
                        inBuffer.clear()

                        val count = minOf(blockSamples, window.samples.size - sampleOffset)
                        val pcmBytes = shortsToLeBytes(window.samples, sampleOffset, count)
                        if (pcmBytes.size > inBuffer.remaining()) {
                            throw IllegalStateException("AAC input buffer zu klein")
                        }
                        inBuffer.put(pcmBytes)

                        val rawPts = (sampleOffset * 1_000_000L / sampleRate)
                        val scaledPts = (rawPts * slowMoFactor).toLong()
                        val ptsUs = if (scaledPts > lastPtsUs) scaledPts else lastPtsUs + 1
                        lastPtsUs = ptsUs

                        codec.queueInputBuffer(inId, 0, pcmBytes.size, ptsUs, 0)
                        sampleOffset += count
                    }
                } else if (!signaledEos) {
                    val inId = codec.dequeueInputBuffer(10_000)
                    if (inId >= 0) {
                        val rawPts = (window.samples.size * 1_000_000L / sampleRate)
                        val scaledPts = (rawPts * slowMoFactor).toLong()
                        val eosPts = max(lastPtsUs + 1, scaledPts)
                        codec.queueInputBuffer(inId, 0, 0, eosPts, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        signaledEos = true
                    }
                }

                when (val outId = codec.dequeueOutputBuffer(info, 10_000)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> outFormat = codec.outputFormat
                    else -> {
                        if (outId >= 0) {
                            val outBuffer = codec.getOutputBuffer(outId)
                                ?: throw IllegalStateException("AAC output buffer null")

                            if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                                info.size = 0
                            }

                            if (info.size > 0) {
                                outBuffer.position(info.offset)
                                outBuffer.limit(info.offset + info.size)
                                val bytes = ByteArray(info.size)
                                outBuffer.get(bytes)
                                samples.add(
                                    EncodedSample(
                                        data = bytes,
                                        ptsUs = info.presentationTimeUs,
                                        flags = info.flags
                                    )
                                )
                            }

                            codec.releaseOutputBuffer(outId, false)
                            if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                        }
                    }
                }
            }
        } finally {
            try {
                codec.stop()
            } catch (_: Throwable) {
            }
            codec.release()
        }

        return EncodedAudio(
            format = outFormat ?: throw IllegalStateException("AAC outputFormat fehlt"),
            samples = samples
        )
    }

    private fun writeAudioSamples(muxer: MediaMuxer, trackIndex: Int, samples: List<EncodedSample>) {
        if (trackIndex < 0) return
        val info = MediaCodec.BufferInfo()
        samples.forEach { sample ->
            val bb = ByteBuffer.wrap(sample.data)
            info.offset = 0
            info.size = sample.data.size
            info.presentationTimeUs = sample.ptsUs
            info.flags = sample.flags
            muxer.writeSampleData(trackIndex, bb, info)
        }
    }

    private fun shortsToLeBytes(src: ShortArray, offset: Int, count: Int): ByteArray {
        val out = ByteArray(count * 2)
        val bb = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until count) bb.putShort(src[offset + i])
        return out
    }

    private fun validateFrames(frames: List<FrameData>, width: Int, height: Int) {
        val expectedLuma = width * height
        frames.forEachIndexed { index, frame ->
            require(frame.width == width && frame.height == height) {
                "Frame $index hat andere Groesse (${frame.width}x${frame.height})"
            }
            require(frame.luma.size >= expectedLuma) {
                "Frame $index luma ungueltig (${frame.luma.size} < $expectedLuma)"
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun selectColorFormat(codecInfo: MediaCodecInfo): Int {
        val caps = codecInfo.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC)
        val available = caps.colorFormats.toSet()

        val preferred = listOf(
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420PackedSemiPlanar,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
        )

        return preferred.firstOrNull { it in available }
            ?: throw IllegalStateException("Kein unterstuetztes YUV420-Format fuer AVC gefunden")
    }

    private fun presentationTimeUs(relativeMs: Long): Long {
        return (relativeMs.coerceAtLeast(0L) * 1000.0 * slowMoFactor).toLong()
    }

    @Suppress("DEPRECATION")
    private fun lumaToYuv420(frame: FrameData, colorFormat: Int): ByteArray {
        val w = frame.width
        val h = frame.height
        val ySize = w * h
        val uvSize = ySize / 4

        val out = ByteArray(ySize + uvSize * 2)
        System.arraycopy(frame.luma, 0, out, 0, ySize)

        when (colorFormat) {
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible -> {
                var uPos = ySize
                var vPos = ySize + uvSize
                repeat(uvSize) {
                    out[uPos++] = 0x80.toByte()
                    out[vPos++] = 0x80.toByte()
                }
            }

            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420PackedSemiPlanar -> {
                var uvPos = ySize
                repeat(uvSize) {
                    out[uvPos++] = 0x80.toByte()
                    out[uvPos++] = 0x80.toByte()
                }
            }

            else -> throw IllegalStateException("Nicht unterstuetztes colorFormat: $colorFormat")
        }

        return out
    }

    private data class EncodedAudio(
        val format: MediaFormat,
        val samples: List<EncodedSample>
    )

    private data class EncodedSample(
        val data: ByteArray,
        val ptsUs: Long,
        val flags: Int
    )

    private data class Mp4EncodingReport(
        val format: String,
        val audioTrackMuxed: Boolean,
        val audioInfo: String?
    )
}
