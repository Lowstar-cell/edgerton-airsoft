package com.example.edgerton_air_soft.trigger

import java.util.concurrent.atomic.AtomicLong

data class TriggerEvent(
    val id: Long,
    val timestampMs: Long,
    val reason: String,
    val motionScore: Double,
    val audioPeakToNoise: Double
)

class TriggerCoordinator(
    private val cooldownMs: Long = 600,
    /** Minimaler Motion-Score damit ein Trigger ausgeloest wird.
     *  Verhindert Phantom-Trigger bei Geraeusch ohne Kamera-Bewegung. */
    var minMotionScore: Double = 0.008,
    /** Minimaler Audio-Peak/Noise Wert fuer einen Trigger. */
    var minAudioPeakToNoise: Double = 3.0
) {
    private val lastTriggerMs = AtomicLong(0L)
    private val counter = AtomicLong(0L)

    /** Anzahl abgelehnter Trigger seit letztem Reset (Debug). */
    @Volatile var rejectedNoMotion: Int = 0; private set
    @Volatile var rejectedCooldown: Int = 0; private set

    fun tryTrigger(
        timestampMs: Long,
        motionScore: Double,
        audioPeakToNoise: Double
    ): TriggerEvent? {
        val last = lastTriggerMs.get()
        if (timestampMs - last < cooldownMs) { rejectedCooldown++; return null }

        // Gate 1: Audio muss stark genug sein (ShotDetectionEngine hat bereits gefiltert,
        // aber hier nochmal als Sicherheit)
        if (audioPeakToNoise < minAudioPeakToNoise) return null

        // Gate 2: Es MUSS Bewegung im Kamerabild geben – sonst ist es kein Schuss
        // sondern nur ein Umgebungsgeraeusch.
        if (motionScore < minMotionScore) { rejectedNoMotion++; return null }

        if (!lastTriggerMs.compareAndSet(last, timestampMs)) return null

        val reason = if (motionScore >= 0.06) "audio+motion" else "audio+low-motion"
        return TriggerEvent(
            id = counter.incrementAndGet(),
            timestampMs = timestampMs,
            reason = reason,
            motionScore = motionScore,
            audioPeakToNoise = audioPeakToNoise
        )
    }

    fun resetStats() { rejectedNoMotion = 0; rejectedCooldown = 0 }
}
