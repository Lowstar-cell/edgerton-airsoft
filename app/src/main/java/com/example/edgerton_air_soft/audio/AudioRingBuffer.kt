package com.example.edgerton_air_soft.audio

import java.util.ArrayDeque
import kotlin.math.max
import kotlin.math.min

class AudioChunk(
    val timestampMs: Long,
    val sampleRate: Int,
    val samples: ShortArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is AudioChunk) return false
        if (timestampMs != other.timestampMs) return false
        if (sampleRate != other.sampleRate) return false
        if (!samples.contentEquals(other.samples)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = timestampMs.hashCode()
        result = 31 * result + sampleRate
        result = 31 * result + samples.contentHashCode()
        return result
    }
}

class AudioWindow(
    val startTimestampMs: Long,
    val sampleRate: Int,
    val samples: ShortArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is AudioWindow) return false
        if (startTimestampMs != other.startTimestampMs) return false
        if (sampleRate != other.sampleRate) return false
        if (!samples.contentEquals(other.samples)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = startTimestampMs.hashCode()
        result = 31 * result + sampleRate
        result = 31 * result + samples.contentHashCode()
        return result
    }
}

class AudioRingBuffer(
    private val retentionMs: Long = 3_000
) {
    private val chunks = ArrayDeque<AudioChunk>()

    @Synchronized
    fun add(chunk: AudioChunk) {
        chunks.addLast(chunk)
        prune(chunk.timestampMs)
    }

    @Synchronized
    fun snapshotWindow(centerTimestampMs: Long, preMs: Long, postMs: Long): AudioWindow? {
        if (chunks.isEmpty()) return null

        val fromMs = centerTimestampMs - preMs
        val toMs = centerTimestampMs + postMs

        val rate = chunks.first().sampleRate
        val selected = chunks.filter { chunk ->
            val durMs = chunkDurationMs(chunk)
            val start = chunk.timestampMs
            val end = chunk.timestampMs + durMs
            end >= fromMs && start <= toMs && chunk.sampleRate == rate
        }
        if (selected.isEmpty()) return null

        val sampleList = ArrayList<Short>(rate * ((preMs + postMs).toInt() / 1000 + 2))
        for (chunk in selected) {
            val durMs = max(1L, chunkDurationMs(chunk))
            val start = chunk.timestampMs
            val end = start + durMs

            val overlapStart = max(start, fromMs)
            val overlapEnd = min(end, toMs)
            if (overlapEnd <= overlapStart) continue

            val fromIndex = ((overlapStart - start) * chunk.samples.size / durMs).toInt().coerceIn(0, chunk.samples.size)
            val toIndex = ((overlapEnd - start) * chunk.samples.size / durMs).toInt().coerceIn(fromIndex, chunk.samples.size)
            for (i in fromIndex until toIndex) {
                sampleList.add(chunk.samples[i])
            }
        }

        if (sampleList.isEmpty()) return null

        val out = ShortArray(sampleList.size)
        for (i in sampleList.indices) out[i] = sampleList[i]
        return AudioWindow(startTimestampMs = fromMs, sampleRate = rate, samples = out)
    }

    @Synchronized
    fun size(): Int = chunks.size

    private fun chunkDurationMs(chunk: AudioChunk): Long {
        return max(1L, chunk.samples.size * 1000L / max(1, chunk.sampleRate))
    }

    private fun prune(nowMs: Long) {
        val minTs = nowMs - retentionMs
        while (chunks.isNotEmpty() && chunks.first().timestampMs < minTs) {
            chunks.removeFirst()
        }
    }
}

