package com.example.edgerton_air_soft.camera

import java.util.ArrayDeque

data class FrameData(
    val timestampMs: Long,
    val width: Int,
    val height: Int,
    val luma: ByteArray,
    /** RGBA Farbdaten (4 Bytes pro Pixel). Nur bei hochwertigem Still-Capture vorhanden. */
    val rgba: ByteArray? = null
)

class FrameRingBuffer(
    private val retentionMs: Long = 2_000,
    private val maxBytes: Long = 48L * 1024L * 1024L,
    private val maxFrames: Int = 180
) {
    private val frames = ArrayDeque<FrameData>()
    private var currentBytes: Long = 0

    @Synchronized
    fun add(frame: FrameData) {
        frames.addLast(frame)
        currentBytes += frame.byteSize()
        prune(frame.timestampMs)
    }

    @Synchronized
    fun snapshotWindow(centerTimestampMs: Long, preMs: Long, postMs: Long): List<FrameData> {
        val from = centerTimestampMs - preMs
        val to = centerTimestampMs + postMs
        return frames.filter { it.timestampMs in from..to }
    }

    @Synchronized
    fun size(): Int = frames.size

    @Synchronized
    fun bytes(): Long = currentBytes

    @Synchronized
    private fun prune(nowMs: Long) {
        val minTs = nowMs - retentionMs
        while (frames.isNotEmpty() && frames.first().timestampMs < minTs) {
            currentBytes -= frames.removeFirst().byteSize()
        }
        while (frames.isNotEmpty() && (currentBytes > maxBytes || frames.size > maxFrames)) {
            currentBytes -= frames.removeFirst().byteSize()
        }
        if (currentBytes < 0L) currentBytes = 0L
    }
}

private fun FrameData.byteSize(): Long = luma.size.toLong() + (rgba?.size?.toLong() ?: 0L)

