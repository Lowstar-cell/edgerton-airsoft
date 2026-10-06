package com.example.edgerton_air_soft.analysis

import com.example.edgerton_air_soft.camera.FrameData
import kotlin.math.abs
import kotlin.math.sqrt

data class TrajectoryPoint(
    val timestampMs: Long,
    val xNorm: Double,
    val yNorm: Double,
    val score: Double
)

data class ImpactHole(
    val xNorm: Double,
    val yNorm: Double,
    val confidence: Double
)

/**
 * Verbesserte Trajektorienerkennung:
 * Statt globalem Centroid aller Differenzpixel werden verbundene Blob-Regionen
 * per Flood-Fill gesucht. Kleine Blobs (BB-groesse) werden bevorzugt und
 * frame-uebergreifend per Nearest-Neighbour verfolgt.
 */
class TrajectoryAnalyzer(
    var diffThreshold: Int = 30,
    private val minBlobPx: Int = 3,
    private val maxBlobPx: Int = 300
) {
    private data class BlobCandidate(val cx: Double, val cy: Double, val area: Int)

    fun analyze(frames: List<FrameData>): List<TrajectoryPoint> {
        if (frames.size < 2) return emptyList()
        val result = mutableListOf<TrajectoryPoint>()
        var prevBlobs = emptyList<BlobCandidate>()
        val step = 2

        for (i in 1 until frames.size) {
            val prev = frames[i - 1]
            val curr = frames[i]
            if (prev.luma.size != curr.luma.size || prev.width != curr.width || curr.width < 4) {
                prevBlobs = emptyList()
                continue
            }

            val blobs = findBlobs(prev.luma, curr.luma, curr.width, curr.height, step)

            // Temporales Tracking: Blobs die nahe an vorherigen Positionen sind bevorzugen
            val maxMatchDist = maxOf(curr.width, curr.height).toDouble() * 0.25
            val tracked = if (prevBlobs.isEmpty()) {
                blobs
            } else {
                blobs.filter { b ->
                    prevBlobs.any { p ->
                        val dx = b.cx - p.cx
                        val dy = b.cy - p.cy
                        sqrt(dx * dx + dy * dy) <= maxMatchDist
                    }
                }.ifEmpty { blobs }
            }

            if (tracked.isNotEmpty()) {
                // BB-aehnlichsten Blob waehlen: kleinster, nicht am Bildrand
                val best = tracked
                    .filter { b ->
                        val xn = b.cx / curr.width
                        val yn = b.cy / curr.height
                        xn in 0.04..0.96 && yn in 0.04..0.96
                    }
                    .minByOrNull { it.area }
                    ?: tracked.minByOrNull { it.area }!!

                // Score: kleiner Blob = hoehere BB-Wahrscheinlichkeit
                val normalizedArea = (best.area.toDouble() / maxBlobPx.toDouble()).coerceIn(0.0, 1.0)
                val score = (1.0 - normalizedArea * 0.7).coerceIn(0.08, 0.95)

                result += TrajectoryPoint(
                    timestampMs = curr.timestampMs,
                    xNorm = (best.cx / curr.width).coerceIn(0.0, 1.0),
                    yNorm = (best.cy / curr.height).coerceIn(0.0, 1.0),
                    score = score
                )
                prevBlobs = tracked
            } else {
                prevBlobs = emptyList()
            }
        }
        return result
    }

    private fun findBlobs(
        prev: ByteArray, curr: ByteArray,
        w: Int, h: Int, step: Int
    ): List<BlobCandidate> {
        val sw = w / step
        val sh = h / step
        if (sw < 2 || sh < 2) return emptyList()
        val total = sw * sh
        val mask = BooleanArray(total)
        val visited = BooleanArray(total)

        // Differenz-Maske aufbauen
        for (y in 0 until sh) {
            for (x in 0 until sw) {
                val srcIdx = (y * step) * w + (x * step)
                if (srcIdx < 0 || srcIdx >= prev.size || srcIdx >= curr.size) continue
                val d = abs((curr[srcIdx].toInt() and 0xFF) - (prev[srcIdx].toInt() and 0xFF))
                mask[y * sw + x] = d >= diffThreshold
            }
        }

        val blobs = mutableListOf<BlobCandidate>()
        val queue = IntArray(total)

        for (startIdx in mask.indices) {
            if (!mask[startIdx] || visited[startIdx]) continue

            var head = 0; var tail = 0
            queue[tail++] = startIdx
            visited[startIdx] = true
            var sumX = 0L; var sumY = 0L; var count = 0

            while (head < tail) {
                val idx = queue[head++]
                val x = idx % sw
                val y = idx / sw
                sumX += x; sumY += y; count++

                val l = if (x > 0) idx - 1 else -1
                val r = if (x < sw - 1) idx + 1 else -1
                val u = if (y > 0) idx - sw else -1
                val d = if (y < sh - 1) idx + sw else -1
                if (l >= 0 && mask[l] && !visited[l]) { visited[l] = true; queue[tail++] = l }
                if (r >= 0 && mask[r] && !visited[r]) { visited[r] = true; queue[tail++] = r }
                if (u >= 0 && mask[u] && !visited[u]) { visited[u] = true; queue[tail++] = u }
                if (d >= 0 && mask[d] && !visited[d]) { visited[d] = true; queue[tail++] = d }
            }

            if (count in minBlobPx..maxBlobPx) {
                blobs += BlobCandidate(
                    cx = (sumX.toDouble() / count) * step,
                    cy = (sumY.toDouble() / count) * step,
                    area = count
                )
            }
        }
        return blobs
    }

    fun detectImpactHole(frames: List<FrameData>): ImpactHole? {
        if (frames.size < 4) return null
        // Vergleiche Bild VOR dem Schuss (erstes Drittel) mit NACH dem Schuss (letztes Drittel)
        // um Kamera-Rauschen und Bewegungs-Artefakte zu reduzieren.
        val beforeIdx = frames.size / 4
        val afterIdx = (frames.size * 3) / 4
        val first = frames[beforeIdx]
        val last = frames[afterIdx]
        if (first.width != last.width || first.height != last.height ||
            first.luma.size != last.luma.size) return null

        val width = first.width
        val height = first.height
        val impactThreshold = diffThreshold + 10  // Hoeher als Tracking-Threshold

        // Phase 1: Finde alle Pixel mit signifikanter Aenderung (nur Abdunkelung = Einschlag)
        var sumW = 0.0; var sumX = 0.0; var sumY = 0.0; var peak = 0.0
        var changedPixels = 0

        var y = 0
        while (y < height) {
            var x = 0
            while (x < width) {
                val idx = y * width + x
                val before = first.luma[idx].toInt() and 0xFF
                val after = last.luma[idx].toInt() and 0xFF
                val delta = before - after  // Positiv = dunkler geworden (Einschlag)
                if (delta > impactThreshold) {
                    val w = delta.toDouble()
                    sumW += w; sumX += x * w; sumY += y * w
                    if (w > peak) peak = w
                    changedPixels++
                }
                x += 2
            }
            y += 2
        }

        if (sumW <= 0.0 || changedPixels < 3) return null

        val centerX = sumX / sumW
        val centerY = sumY / sumW

        // Phase 2: Kompaktheits-Check – ein Einschlag ist lokal konzentriert.
        // Berechne wie viele geaenderte Pixel innerhalb eines kleinen Radius um das Zentrum liegen.
        val maxRadiusPx = maxOf(width, height) * 0.08  // 8% des Bildes
        var nearCount = 0
        y = 0
        while (y < height) {
            var x = 0
            while (x < width) {
                val idx = y * width + x
                val before = first.luma[idx].toInt() and 0xFF
                val after = last.luma[idx].toInt() and 0xFF
                val delta = before - after
                if (delta > impactThreshold) {
                    val dx = x - centerX
                    val dy = y - centerY
                    if (dx * dx + dy * dy <= maxRadiusPx * maxRadiusPx) nearCount++
                }
                x += 2
            }
            y += 2
        }

        // Mindestens 50% der geaenderten Pixel muessen in der Naehe des Zentrums liegen
        val compactness = nearCount.toDouble() / changedPixels.toDouble()
        if (compactness < 0.40) return null

        val xNorm = centerX / width
        val yNorm = centerY / height
        val confidence = ((compactness * 0.5) + (peak / 255.0) * 0.3 +
                (changedPixels.toDouble() / ((width / 4.0) * (height / 4.0))).coerceIn(0.0, 1.0) * 0.2).coerceIn(0.0, 1.0)
        if (confidence < 0.08) return null
        return ImpactHole(xNorm = xNorm, yNorm = yNorm, confidence = confidence)
    }
}

