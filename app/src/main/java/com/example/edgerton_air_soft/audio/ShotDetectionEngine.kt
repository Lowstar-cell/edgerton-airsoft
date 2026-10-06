package com.example.edgerton_air_soft.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Minimaler Echtzeit-Detektor fuer impulsive Schussgeraesche.
 * Ziel: robust genug gegen Gespraech/Musik durch adaptive Schwelle + Impulsmerkmale.
 */
class ShotDetectionEngine(
    initialConfig: DetectorConfig = DetectorConfig()
) {
    /** Live-aenderbar: wird im naechsten Audio-Block wirksam (Thread-safe via @Volatile). */
    @Volatile
    var config: DetectorConfig = initialConfig
        private set

    fun updateConfig(new: DetectorConfig) {
        config = new
    }

    private val executor = Executors.newSingleThreadExecutor()
    private val running = AtomicBoolean(false)

    @Volatile
    private var calibratingUntilNanos: Long = 0L

    @Volatile
    private var state = DetectorState()

    fun start(listener: Listener) {
        if (running.getAndSet(true)) return
        executor.execute { captureLoop(listener) }
    }

    fun stop() {
        running.set(false)
    }

    fun startCalibration(durationMs: Long = config.calibrationDurationMs) {
        calibratingUntilNanos = System.nanoTime() + durationMs * 1_000_000L
    }

    interface Listener {
        fun onStateUpdate(state: DetectorState)
        fun onShotDetected(event: ShotEvent)
        fun onError(message: String)
        fun onAudioChunk(chunk: AudioChunk) {}
    }

    @SuppressLint("MissingPermission")
    private fun captureLoop(listener: Listener) {
        val minBuffer = AudioRecord.getMinBufferSize(
            config.sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        if (minBuffer <= 0) {
            running.set(false)
            listener.onError("Audio init fehlgeschlagen (minBuffer=$minBuffer)")
            return
        }

        val bufferSize = max(minBuffer, config.blockSize * 4)
        val audioRecord = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                config.sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
        } catch (se: SecurityException) {
            running.set(false)
            listener.onError("Keine Mikrofon-Berechtigung")
            return
        }

        if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            running.set(false)
            listener.onError("AudioRecord konnte nicht initialisiert werden")
            return
        }

        val block = ShortArray(config.blockSize)

        var noiseFloor = config.initialNoiseFloor
        var lastShotMs = 0L
        var inAttack = false
        var attackStartMs = 0L

        var prevX = 0.0
        var prevY = 0.0

        try {
            try {
                audioRecord.startRecording()
            } catch (se: SecurityException) {
                running.set(false)
                listener.onError("Mikrofonzugriff verweigert")
                return
            }

            while (running.get()) {
                val read = audioRecord.read(block, 0, block.size)
                if (read <= 0) continue

                val nowMs = SystemClock.elapsedRealtime()
                val blockDurationMs = (read * 1000L / config.sampleRate).coerceAtLeast(1L)
                val chunkStartMs = nowMs - blockDurationMs
                listener.onAudioChunk(
                    AudioChunk(
                        timestampMs = chunkStartMs,
                        sampleRate = config.sampleRate,
                        samples = block.copyOf(read)
                    )
                )

                var sumSq = 0.0
                var hpSumSq = 0.0
                var peak = 0.0

                for (i in 0 until read) {
                    val x = block[i] / 32768.0
                    val absX = abs(x)
                    if (absX > peak) peak = absX
                    sumSq += x * x

                    val y = config.highPassA * (prevY + x - prevX)
                    prevX = x
                    prevY = y
                    hpSumSq += y * y
                }

                val rms = sqrt(sumSq / read)
                val hpRms = sqrt(hpSumSq / read)
                val hpRatio = hpRms / max(rms, 1e-7)
                val crest = peak / max(rms, 1e-7)
                val isCalibrating = System.nanoTime() < calibratingUntilNanos

                val alpha = if (isCalibrating) config.calibrationAlpha else config.noiseAlpha
                val maxNoiseUpdate = noiseFloor * config.noiseUpdateCeilingMultiplier
                if (rms < maxNoiseUpdate || isCalibrating) {
                    noiseFloor = (1.0 - alpha) * noiseFloor + alpha * rms
                }

                val peakToNoise = peak / max(noiseFloor, 1e-7)
                val dynamicGate = max(config.rmsGate, noiseFloor * config.noiseGateMultiplier)
                val candidate =
                    rms > dynamicGate &&
                        peakToNoise > config.peakToNoiseThreshold &&
                        crest > config.crestThreshold &&
                        hpRatio > config.highPassRatioThreshold

                if (candidate && !inAttack) {
                    inAttack = true
                    attackStartMs = nowMs
                }
                if (!candidate) {
                    inAttack = false
                }

                val cooldownOver = nowMs - lastShotMs >= config.cooldownMs
                val attackHeld = inAttack && (nowMs - attackStartMs >= config.attackHoldMs)
                if (attackHeld && cooldownOver) {
                    lastShotMs = nowMs
                    inAttack = false
                    listener.onShotDetected(
                        ShotEvent(
                            timestampMs = nowMs,
                            rms = rms,
                            peak = peak,
                            peakToNoise = peakToNoise,
                            crest = crest,
                            highPassRatio = hpRatio
                        )
                    )
                }

                state = DetectorState(
                    isRunning = true,
                    isCalibrating = isCalibrating,
                    noiseFloor = noiseFloor,
                    rms = rms,
                    peak = peak,
                    peakToNoise = peakToNoise,
                    crest = crest,
                    highPassRatio = hpRatio,
                    dynamicGate = dynamicGate
                )
                listener.onStateUpdate(state)
            }
        } catch (t: Throwable) {
            listener.onError("Audio loop Fehler: ${t.message ?: t::class.java.simpleName}")
        } finally {
            try {
                audioRecord.stop()
            } catch (_: IllegalStateException) {
            }
            audioRecord.release()
            running.set(false)
            listener.onStateUpdate(state.copy(isRunning = false, isCalibrating = false))
        }
    }
}

data class DetectorConfig(
    val sampleRate: Int = 16_000,
    val blockSize: Int = 512,
    val cooldownMs: Long = 350,
    val attackHoldMs: Long = 12,
    val calibrationDurationMs: Long = 2_000,
    val initialNoiseFloor: Double = 0.005,
    val calibrationAlpha: Double = 0.2,
    val noiseAlpha: Double = 0.015,
    val noiseGateMultiplier: Double = 2.8,
    val noiseUpdateCeilingMultiplier: Double = 1.6,
    val rmsGate: Double = 0.012,
    val peakToNoiseThreshold: Double = 3.5,
    val crestThreshold: Double = 2.5,
    val highPassRatioThreshold: Double = 0.50,
    val highPassA: Double = 0.95
)

data class DetectorState(
    val isRunning: Boolean = false,
    val isCalibrating: Boolean = false,
    val noiseFloor: Double = 0.0,
    val rms: Double = 0.0,
    val peak: Double = 0.0,
    val peakToNoise: Double = 0.0,
    val crest: Double = 0.0,
    val highPassRatio: Double = 0.0,
    val dynamicGate: Double = 0.0
)

data class ShotEvent(
    val timestampMs: Long,
    val rms: Double,
    val peak: Double,
    val peakToNoise: Double,
    val crest: Double,
    val highPassRatio: Double
)

// AudioChunk ist in AudioRingBuffer.kt definiert.
