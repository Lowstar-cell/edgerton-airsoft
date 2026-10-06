package com.example.edgerton_air_soft.motion

import com.example.edgerton_air_soft.camera.FrameData
import kotlin.math.abs

data class MotionState(
    val score: Double = 0.0,
    val confidence: Double = 0.0,
    val maxDiffNorm: Double = 0.0,
    val yoloConfidence: Double = 0.0,
    val source: String = "diff"
)

class MotionAnalyzer(
    private val sampleStep: Int = 8,
    private val alpha: Double = 0.25,
    private val triggerDiffNorm: Double = 0.18,
    /** Mindest-Anteil geaenderter Pixel fuer "echte" Bewegung (vs. Rauschen). */
    private val minChangedFraction: Double = 0.003,
    private val noisePixelThreshold: Double = 0.06
) {
    private var previous: ByteArray? = null
    private var scratch: ByteArray? = null
    private var filteredScore = 0.0

    fun analyze(frame: FrameData): MotionState {
        val prev = previous
        if (prev == null || prev.size != frame.luma.size) {
            val fresh = ByteArray(frame.luma.size)
            System.arraycopy(frame.luma, 0, fresh, 0, frame.luma.size)
            previous = fresh
            scratch = ByteArray(frame.luma.size)
            return MotionState()
        }

        var sum = 0.0
        var maxDiff = 0.0
        var count = 0
        var changedCount = 0
        var i = 0
        while (i < frame.luma.size) {
            val current = frame.luma[i].toInt() and 0xFF
            val old = prev[i].toInt() and 0xFF
            val diff = abs(current - old) / 255.0
            sum += diff
            if (diff > maxDiff) maxDiff = diff
            if (diff > noisePixelThreshold) changedCount++
            count++
            i += sampleStep
        }

        val instant = if (count == 0) 0.0 else sum / count
        filteredScore = (1.0 - alpha) * filteredScore + alpha * instant
        val work = scratch?.takeIf { it.size == frame.luma.size } ?: ByteArray(frame.luma.size).also { scratch = it }
        System.arraycopy(frame.luma, 0, work, 0, frame.luma.size)
        scratch = prev
        previous = work

        val diffConfidence = (filteredScore / triggerDiffNorm).coerceIn(0.0, 1.0)

        // Raeuml. Bewegungsanteil: Wie viele Pixel haben sich signifikant geaendert?
        val changedFraction = if (count == 0) 0.0 else changedCount.toDouble() / count
        val spatialConfidence = (changedFraction / minChangedFraction).coerceIn(0.0, 1.0)

        // Kombiniert: Sowohl durchschnittliche Aenderung ALS AUCH raeumliche Verteilung
        val fusedConfidence = (diffConfidence * 0.6 + spatialConfidence * 0.4).coerceIn(0.0, 1.0)
        val source = "diff"

        return MotionState(
            score = filteredScore,
            confidence = fusedConfidence,
            maxDiffNorm = maxDiff,
            yoloConfidence = spatialConfidence,
            source = source
        )
    }
}

