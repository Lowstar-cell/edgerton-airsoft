package com.example.edgerton_air_soft

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.app.ActivityManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.core.graphics.createBitmap
import android.net.Uri
import android.widget.MediaController
import android.widget.VideoView
import androidx.core.net.toUri
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.PrimaryScrollableTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.edgerton_air_soft.analysis.TrajectoryAnalyzer
import com.example.edgerton_air_soft.audio.AudioChunk
import com.example.edgerton_air_soft.audio.AudioRingBuffer
import com.example.edgerton_air_soft.audio.DetectorState
import com.example.edgerton_air_soft.audio.ShotDetectionEngine
import com.example.edgerton_air_soft.audio.ShotEvent
import com.example.edgerton_air_soft.camera.CameraSessionController
import com.example.edgerton_air_soft.camera.FrameData
import com.example.edgerton_air_soft.camera.FrameRingBuffer
import com.example.edgerton_air_soft.export.FallbackReplayExporter
import com.example.edgerton_air_soft.export.ImageSequenceExporter
import com.example.edgerton_air_soft.export.Mp4SlowMotionExporter
import com.example.edgerton_air_soft.motion.MotionAnalyzer
import com.example.edgerton_air_soft.motion.MotionState
import com.example.edgerton_air_soft.sam.SamOnnxHelper
import com.example.edgerton_air_soft.segmentation.LocalSegmenter
import com.example.edgerton_air_soft.segmentation.TargetEntry
import com.example.edgerton_air_soft.trigger.TriggerCoordinator
import com.example.edgerton_air_soft.ui.overlay.DetectionOverlay
import com.example.edgerton_air_soft.ui.overlay.OverlayViewState
import com.example.edgerton_air_soft.ui.theme.EdgertonAirsoftTheme
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicReference
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject

internal data class ShotRecord(
    val playerName: String,
    val timestampMs: Long,
    val xNorm: Double?,
    val yNorm: Double?,
    val awarded: Int,
    val reason: String,
    val roundElapsedMs: Long
)

internal data class WeaponProfile(
    val id: String,
    val name: String,
    val photoUri: String = "",
    val fps: Int = 0,
    val bbWeightG: Double = 0.20,
    val hopUp: Int = 0,
    val notes: String = ""
)

private fun weaponProfilesToJson(profiles: List<WeaponProfile>): String {
    val arr = JSONArray()
    profiles.forEach { p ->
        arr.put(
            JSONObject()
                .put("id", p.id)
                .put("name", p.name)
                .put("photoUri", p.photoUri)
                .put("fps", p.fps)
                .put("bbWeightG", p.bbWeightG)
                .put("hopUp", p.hopUp)
                .put("notes", p.notes)
        )
    }
    return arr.toString()
}

private fun weaponProfilesFromJson(raw: String?): List<WeaponProfile> {
    if (raw.isNullOrBlank()) return emptyList()
    return try {
        val arr = JSONArray(raw)
        buildList {
            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                add(
                    WeaponProfile(
                        id = o.optString("id", "w$i"),
                        name = o.optString("name", "Waffe ${i + 1}"),
                        photoUri = o.optString("photoUri", ""),
                        fps = o.optInt("fps", 0),
                        bbWeightG = o.optDouble("bbWeightG", 0.20),
                        hopUp = o.optInt("hopUp", 0),
                        notes = o.optString("notes", "")
                    )
                )
            }
        }
    } catch (_: Throwable) {
        emptyList()
    }
}

/** Schnelle reine Kotlin-Segmentierung – kein ML, kein OpenCV, sofortig. */
private fun detectBoardFast(frame: FrameData): SegmentationAttempt {
    val rgba = frame.rgba
    val w = frame.width
    val h = frame.height
    if (w < 32 || h < 32) return SegmentationAttempt(null, "fast: frame zu klein")

    val maxDim = 320
    val step = maxOf(1, maxOf(w, h) / maxDim)
    val sw = maxOf(8, w / step)
    val sh = maxOf(8, h / step)
    val bright = IntArray(sw * sh)
    val sat = IntArray(sw * sh)

    var p = 0
    for (y in 0 until sh) {
        val srcY = minOf(h - 1, y * step)
        for (x in 0 until sw) {
            val srcX = minOf(w - 1, x * step)
            val r: Int
            val g: Int
            val b: Int
            if (rgba != null) {
                val pi = (srcY * w + srcX) * 4
                r = (rgba.getOrNull(pi)?.toInt() ?: 128) and 0xFF
                g = (rgba.getOrNull(pi + 1)?.toInt() ?: 128) and 0xFF
                b = (rgba.getOrNull(pi + 2)?.toInt() ?: 128) and 0xFF
            } else {
                val lv = frame.luma.getOrNull(srcY * w + srcX)?.toInt()?.and(0xFF) ?: 128
                r = lv; g = lv; b = lv
            }
            val maxC = maxOf(r, g, b)
            val minC = minOf(r, g, b)
            bright[p] = maxC
            sat[p] = if (maxC == 0) 0 else ((maxC - minC) * 255 / maxC)
            p++
        }
    }

    val candidates = mutableListOf<Pair<SegmentRegion, String>>()

    fun tryMask(mask: BooleanArray, label: String) {
        val dil = BooleanArray(sw * sh)
        for (y in 0 until sh) for (x in 0 until sw) {
            if (!mask[y * sw + x]) continue
            for (dy in -2..2) for (dx in -2..2) {
                val nx = x + dx
                val ny = y + dy
                if (nx in 0 until sw && ny in 0 until sh) dil[ny * sw + nx] = true
            }
        }
        val closed = BooleanArray(sw * sh) { i ->
            if (!dil[i]) return@BooleanArray false
            val x = i % sw
            val y = i / sw
            for (dy in -2..2) for (dx in -2..2) {
                val nx = x + dx
                val ny = y + dy
                if (nx !in 0 until sw || ny !in 0 until sh || !dil[ny * sw + nx]) return@BooleanArray false
            }
            true
        }
        val region = largestComponent(closed, sw, sh) ?: return
        val bw = (region.maxX - region.minX + 1).coerceAtLeast(1)
        val bh = (region.maxY - region.minY + 1).coerceAtLeast(1)
        val fill = region.area.toDouble() / (bw * bh)
        val areaNorm = bw.toDouble() * bh / (sw * sh)
        if (areaNorm !in 0.03..0.92 || fill < 0.40) return
        candidates.add(region to label)
    }

    tryMask(BooleanArray(sw * sh) { i -> bright[i] >= 160 && sat[i] < 70 }, "weiss-papier")
    val brightSorted = bright.copyOf().also { it.sort() }
    val brightThresh = brightSorted[((sw * sh) * 0.70).toInt().coerceIn(0, sw * sh - 1)]
    tryMask(BooleanArray(sw * sh) { i -> bright[i] >= brightThresh }, "hell-top30")
    val satMedian = sat.copyOf().also { it.sort() }[sat.size / 2]
    tryMask(BooleanArray(sw * sh) { i -> sat[i] < maxOf(satMedian, 35) && bright[i] >= 80 }, "entsaettigt")

    if (candidates.isEmpty()) return SegmentationAttempt(null, "fast: keine Region")

    val (region, label) = candidates.maxByOrNull { it.first.area } ?: return SegmentationAttempt(null, "fast: kein Kandidat")
    val x0 = (region.minX.toDouble() / sw).coerceIn(0.0, 1.0)
    val y0 = (region.minY.toDouble() / sh).coerceIn(0.0, 1.0)
    val x1 = ((region.maxX + 1).toDouble() / sw).coerceIn(0.0, 1.0)
    val y1 = ((region.maxY + 1).toDouble() / sh).coerceIn(0.0, 1.0)
    val bw = (region.maxX - region.minX + 1).coerceAtLeast(1)
    val bh = (region.maxY - region.minY + 1).coerceAtLeast(1)
    val fill = region.area.toDouble() / (bw * bh)
    val areaNorm = ((x1 - x0) * (y1 - y0)).coerceIn(0.0, 1.0)
    val conf = (fill * 0.55 + (1.0 - kotlin.math.abs(areaNorm - 0.25)) * 0.45).coerceIn(0.0, 1.0)
    val result = AutoBoardResult(
        tlX = x0,
        tlY = y0,
        trX = x1,
        trY = y0,
        brX = x1,
        brY = y1,
        blX = x0,
        blY = y1,
        confidence = conf,
        segmentedFill = fill,
        segmentedAreaNorm = areaNorm
    )
    return SegmentationAttempt(result, "fast:$label conf=${format3(conf)}")
}

/** Segmenter: OpenCV-FloodFill (primaer, gleicher Pfad wie QmStone) → Kotlin-Heuristik */
private class HybridSegmenter(private val context: Context) {

    fun detectAutomatic(frame: FrameData, pointHint: Pair<Double, Double>? = null): SegmentationAttempt {
        if (frame.rgba == null && frame.luma.isEmpty()) return SegmentationAttempt(null, "kein Bild")

        val bmp: android.graphics.Bitmap? = if (frame.rgba != null)
            convertRgbaToBitmap(frame.rgba, frame.width, frame.height) else null

        if (bmp != null) {
            val pxNorm = pointHint?.first ?: 0.5
            val pyNorm = pointHint?.second ?: 0.5

            // ── ML-Pfad: EfficientSAM (Primaer wenn verfuegbar) ─────────────
            if (SamOnnxHelper.isEfficientSamAvailable()) {
                try {
                    val samMask = SamOnnxHelper.runEfficientSamPoint(bmp, pxNorm, pyNorm)
                    if (samMask != null) {
                        val sr = maskToRegion(samMask)
                        if (sr != null) {
                            val tapInfo = if (pointHint != null)
                                "-sam-tap(${String.format(Locale.US, "%.2f", pxNorm)},${String.format(Locale.US, "%.2f", pyNorm)})"
                                else "-sam-auto"
                            
                            val result = segmentRegionToAutoBoardResult(sr, samMask.width, samMask.height)
                            bmp.recycle()
                            return SegmentationAttempt(result, "esam$tapInfo", samMask)
                        }
                        samMask.recycle()
                    }
                } catch (t: Throwable) {
                    android.util.Log.e("HybridSegmenter", "SAM failed, falling back: ${t.message}")
                }
            }

            // ── OpenCV FloodFill – FALLBACK ───────────────────────────
            try {
                val sx = (pxNorm * bmp.width).toInt().coerceIn(0, bmp.width - 1)
                val sy = (pyNorm * bmp.height).toInt().coerceIn(0, bmp.height - 1)

                val segResult = kotlinx.coroutines.runBlocking {
                    LocalSegmenter.segmentFromPoint(bmp, sx, sy)
                }

                val sr = maskToRegion(segResult.maskBitmap)
                // Mindestflaeche: 0.1% des Bildes, sonst war's ein Mis-Tap auf einem
                // verrauschten Pixel und die Maske ist Mini-Mini.
                val minPx = (segResult.maskBitmap.width.toLong() * segResult.maskBitmap.height.toLong() / 1000L).toInt()
                if (sr != null && segResult.pixelArea >= minPx) {
                    val tapInfo = if (pointHint != null)
                        "-tap(${String.format(Locale.US, "%.2f", pxNorm)},${String.format(Locale.US, "%.2f", pyNorm)})"
                        else ""
                    bmp.recycle()
                    return SegmentationAttempt(
                        segmentRegionToAutoBoardResult(sr, segResult.maskBitmap.width, segResult.maskBitmap.height),
                        "floodfill$tapInfo",
                        segResult.maskBitmap
                    )
                }
                segResult.maskBitmap.recycle()
            } catch (t: Throwable) {
                android.util.Log.e("HybridSegmenter", "LocalSegmenter failed: ${t.message}", t)
            }
        }

        bmp?.recycle()

        // ── Kotlin-Heuristik – letzter Ausweg ───────────────────────────
        val fast = detectBoardFast(frame)
        return if (fast.result != null) fast
        else SegmentationAttempt(null, "kein Ergebnis")
    }

    fun close() { /* No-Op – LocalSegmenter haelt keinen State zum freigeben. */ }

    fun maskToRegion(mask: Bitmap): SegmentRegion? {
        val w = mask.width; val h = mask.height
        val pixels = IntArray(w * h); mask.getPixels(pixels, 0, w, 0, 0, w, h)
        val boolMask = BooleanArray(w * h) { i -> ((pixels[i] ushr 24) and 0xFF) > 16 }
        return largestComponent(boolMask, w, h)
    }

    fun segmentRegionToAutoBoardResult(region: SegmentRegion, sw: Int, sh: Int): AutoBoardResult {
        val x0 = (region.minX.toDouble() / sw).coerceIn(0.0, 1.0)
        val y0 = (region.minY.toDouble() / sh).coerceIn(0.0, 1.0)
        val x1 = ((region.maxX + 1).toDouble() / sw).coerceIn(0.0, 1.0)
        val y1 = ((region.maxY + 1).toDouble() / sh).coerceIn(0.0, 1.0)
        val areaNorm = ((x1 - x0) * (y1 - y0)).coerceIn(0.0, 1.0)
        val bboxW = (region.maxX - region.minX + 1).coerceAtLeast(1)
        val bboxH = (region.maxY - region.minY + 1).coerceAtLeast(1)
        val fill = region.area.toDouble() / (bboxW * bboxH)
        val confidence = ((region.area.toDouble() / (sw * sh)).coerceIn(0.0, 1.0) * 0.6 + 0.4 * (1.0 - kotlin.math.abs(areaNorm - 0.25))).coerceIn(0.0, 1.0)
        return AutoBoardResult(tlX = x0, tlY = y0, trX = x1, trY = y0, brX = x1, brY = y1, blX = x0, blY = y1, confidence = confidence, segmentedFill = fill, segmentedAreaNorm = areaNorm)
    }

    private fun convertRgbaToBitmap(rgba: ByteArray, w: Int, h: Int): Bitmap {
        val pixels = IntArray(w * h)
        for (i in pixels.indices) {
            val j = i * 4
            val r = rgba.getOrNull(j)?.toInt() ?: 128
            val g = rgba.getOrNull(j + 1)?.toInt() ?: 128
            val b = rgba.getOrNull(j + 2)?.toInt() ?: 128
            val a = rgba.getOrNull(j + 3)?.toInt() ?: 255
            pixels[i] = (a and 0xFF shl 24) or ((r and 0xFF) shl 16) or ((g and 0xFF) shl 8) or (b and 0xFF)
        }
        val bmp = createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return bmp
    }
}

private data class AutoBoardResult(
    val tlX: Double,
    val tlY: Double,
    val trX: Double,
    val trY: Double,
    val brX: Double,
    val brY: Double,
    val blX: Double,
    val blY: Double,
    val confidence: Double,
    val segmentedFill: Double,
    val segmentedAreaNorm: Double
)

private data class SegmentationAttempt(
    val result: AutoBoardResult?,
    val debug: String,
    val mask: android.graphics.Bitmap? = null
)

private data class SegmentRegion(
    val minX: Int,
    val minY: Int,
    val maxX: Int,
    val maxY: Int,
    val area: Int
)

private fun rectIou(a: AutoBoardResult, b: AutoBoardResult): Double {
    val ax0 = minOf(a.tlX, a.blX)
    val ay0 = minOf(a.tlY, a.trY)
    val ax1 = maxOf(a.trX, a.brX)
    val ay1 = maxOf(a.blY, a.brY)
    val bx0 = minOf(b.tlX, b.blX)
    val by0 = minOf(b.tlY, b.trY)
    val bx1 = maxOf(b.trX, b.brX)
    val by1 = maxOf(b.blY, b.brY)
    val ix0 = maxOf(ax0, bx0)
    val iy0 = maxOf(ay0, by0)
    val ix1 = minOf(ax1, bx1)
    val iy1 = minOf(ay1, by1)
    val iw = (ix1 - ix0).coerceAtLeast(0.0)
    val ih = (iy1 - iy0).coerceAtLeast(0.0)
    val inter = iw * ih
    val areaA = (ax1 - ax0).coerceAtLeast(0.0) * (ay1 - ay0).coerceAtLeast(0.0)
    val areaB = (bx1 - bx0).coerceAtLeast(0.0) * (by1 - by0).coerceAtLeast(0.0)
    val union = areaA + areaB - inter
    return if (union <= 1e-9) 0.0 else (inter / union).coerceIn(0.0, 1.0)
}

private fun buildFullImageScanPoints(width: Int, height: Int): List<Pair<Double, Double>> {
    if (width <= 0 || height <= 0) return listOf(0.5 to 0.5)
    val aspect = width.toDouble() / height.toDouble()
    val cols = when {
        aspect >= 1.6 -> 9
        aspect >= 1.2 -> 8
        aspect >= 0.85 -> 7
        else -> 6
    }
    val rows = when {
        aspect >= 1.6 -> 6
        aspect >= 1.2 -> 7
        aspect >= 0.85 -> 8
        else -> 9
    }
    val marginX = 0.04
    val marginY = 0.04
    val usableW = 1.0 - marginX * 2.0
    val usableH = 1.0 - marginY * 2.0
    val stepX = usableW / cols
    val stepY = usableH / rows
    val points = LinkedHashSet<Pair<Double, Double>>()

    fun addPass(offsetX: Double, offsetY: Double) {
        for (row in 0 until rows) {
            for (col in 0 until cols) {
                val x = (marginX + (col + offsetX) * stepX).coerceIn(0.02, 0.98)
                val y = (marginY + (row + offsetY) * stepY).coerceIn(0.02, 0.98)
                points += x to y
            }
        }
    }

    addPass(0.5, 0.5)
    addPass(0.25, 0.25)
    addPass(0.75, 0.75)
    points += 0.5 to 0.5
    return points.toList()
}

private fun largestComponent(mask: BooleanArray, w: Int, h: Int): SegmentRegion? {
    if (mask.isEmpty()) return null
    val visited = BooleanArray(mask.size)
    val queue = IntArray(mask.size)
    var best: SegmentRegion? = null

    for (i in mask.indices) {
        if (!mask[i] || visited[i]) continue
        var head = 0
        var tail = 0
        queue[tail++] = i
        visited[i] = true

        var minX = Int.MAX_VALUE
        var minY = Int.MAX_VALUE
        var maxX = Int.MIN_VALUE
        var maxY = Int.MIN_VALUE
        var area = 0

        while (head < tail) {
            val idx = queue[head++]
            val x = idx % w
            val y = idx / w
            area++
            if (x < minX) minX = x
            if (y < minY) minY = y
            if (x > maxX) maxX = x
            if (y > maxY) maxY = y

            val left = if (x > 0) idx - 1 else -1
            val right = if (x < w - 1) idx + 1 else -1
            val up = if (y > 0) idx - w else -1
            val down = if (y < h - 1) idx + w else -1

            if (left >= 0 && mask[left] && !visited[left]) {
                visited[left] = true
                queue[tail++] = left
            }
            if (right >= 0 && mask[right] && !visited[right]) {
                visited[right] = true
                queue[tail++] = right
            }
            if (up >= 0 && mask[up] && !visited[up]) {
                visited[up] = true
                queue[tail++] = up
            }
            if (down >= 0 && mask[down] && !visited[down]) {
                visited[down] = true
                queue[tail++] = down
            }
        }

        val region = SegmentRegion(minX = minX, minY = minY, maxX = maxX, maxY = maxY, area = area)
        if (best == null || region.area > best.area) {
            best = region
        }
    }

    return best
}

// ── Otsu Threshold ──────────────────────────────────────────────────
private fun otsuThreshold(gray: IntArray): Int {
    val histogram = IntArray(256)
    for (v in gray) histogram[v.coerceIn(0, 255)]++
    val total = gray.size.toDouble()
    var sumAll = 0.0
    for (i in 0..255) sumAll += i * histogram[i]
    var sumB = 0.0
    var wB = 0.0
    var best = 0
    var maxVariance = 0.0
    for (t in 0..255) {
        wB += histogram[t]
        if (wB <= 0) continue
        val wF = total - wB
        if (wF <= 0) break
        sumB += t * histogram[t]
        val mB = sumB / wB
        val mF = (sumAll - sumB) / wF
        val diff = mB - mF
        val variance = wB * wF * diff * diff
        if (variance > maxVariance) { maxVariance = variance; best = t }
    }
    return best
}

// ── Morphologische Operatoren ────────────────────────────────────────────
private fun morphErode(mask: BooleanArray, w: Int, h: Int, r: Int): BooleanArray {
    val out = BooleanArray(w * h)
    for (y in 0 until h) for (x in 0 until w) {
        if (!mask[y * w + x]) continue
        var ok = true
        outer@ for (dy in -r..r) for (dx in -r..r) {
            val nx = x + dx; val ny = y + dy
            if (nx !in 0 until w || ny !in 0 until h || !mask[ny * w + nx]) { ok = false; break@outer }
        }
        out[y * w + x] = ok
    }
    return out
}

private fun morphDilate(mask: BooleanArray, w: Int, h: Int, r: Int): BooleanArray {
    val out = BooleanArray(w * h)
    for (y in 0 until h) for (x in 0 until w) {
        if (!mask[y * w + x]) continue
        for (dy in -r..r) for (dx in -r..r) {
            val nx = x + dx; val ny = y + dy
            if (nx in 0 until w && ny in 0 until h) out[ny * w + nx] = true
        }
    }
    return out
}

private fun morphOpen(mask: BooleanArray, w: Int, h: Int, r: Int) =
    morphDilate(morphErode(mask, w, h, r), w, h, r)

private fun morphClose(mask: BooleanArray, w: Int, h: Int, r: Int) =
    morphErode(morphDilate(mask, w, h, r), w, h, r)

/** Konvertiert FrameData (RGBA) zu einem Bitmap für die Anzeige und SAM-Inferenz. */
private fun convertFrameToBitmap(frame: FrameData): Bitmap? {
    val rgba = frame.rgba ?: return null
    val w = frame.width; val h = frame.height
    if (w <= 0 || h <= 0 || rgba.size < w * h * 4) return null
    val pixels = IntArray(w * h)
    for (i in pixels.indices) {
        val j = i * 4
        pixels[i] = ((rgba[j + 3].toInt() and 0xFF) shl 24) or
                    ((rgba[j].toInt() and 0xFF) shl 16) or
                    ((rgba[j + 1].toInt() and 0xFF) shl 8) or
                    (rgba[j + 2].toInt() and 0xFF)
    }
    return createBitmap(w, h, Bitmap.Config.ARGB_8888).also { it.setPixels(pixels, 0, w, 0, 0, w, h) }
}

/**
 * Hochqualitaets-Segmentierung auf einem einzelnen Standbild.
 * Skaliert auf max 1080px (statt 320px bei detectBoardFast).
 * Otsu-Schwelle + morphologisches Open/Close + konvexe Eckensuche.
 * Liefert skalpel-scharfe Kanten auch bei weissem Blatt auf dunkler Wand.
 */
private fun detectBoardHighQuality(frame: FrameData): SegmentationAttempt {
    val w = frame.width; val h = frame.height
    if (w < 32 || h < 32) return SegmentationAttempt(null, "hq: frame zu klein")

    val maxDim = 1080
    val step = maxOf(1, maxOf(w, h) / maxDim)
    val sw = (w / step).coerceAtLeast(8)
    val sh = (h / step).coerceAtLeast(8)

    val bright = IntArray(sw * sh)
    for (y in 0 until sh) {
        val srcY = minOf(h - 1, y * step)
        for (x in 0 until sw) {
            val srcX = minOf(w - 1, x * step)
            bright[y * sw + x] = if (frame.rgba != null) {
                val pi = (srcY * w + srcX) * 4
                val r = (frame.rgba.getOrNull(pi)?.toInt() ?: 128) and 0xFF
                val g = (frame.rgba.getOrNull(pi + 1)?.toInt() ?: 128) and 0xFF
                val b = (frame.rgba.getOrNull(pi + 2)?.toInt() ?: 128) and 0xFF
                maxOf(r, g, b)
            } else {
                (frame.luma.getOrNull(srcY * w + srcX)?.toInt() ?: 128) and 0xFF
            }
        }
    }

    // Otsu: optimale Schwelle fuer bimodales Histogramm (Papier vs. Hintergrund)
    val thresh = otsuThreshold(bright)
    val raw = BooleanArray(sw * sh) { i -> bright[i] > thresh }

    // Adaptiver Morphologie-Radius
    val r1 = maxOf(1, minOf(sw, sh) / 120)
    val r2 = maxOf(2, minOf(sw, sh) / 60)
    val processed = morphClose(morphOpen(raw, sw, sh, r1), sw, sh, r2)

    val region = largestComponent(processed, sw, sh)
        ?: return SegmentationAttempt(null, "hq: keine Region nach Morphologie")

    val bw2 = (region.maxX - region.minX + 1).coerceAtLeast(1)
    val bh2 = (region.maxY - region.minY + 1).coerceAtLeast(1)
    val fill = region.area.toDouble() / (bw2 * bh2)
    val areaNorm = bw2.toDouble() * bh2 / (sw * sh)

    // Deutlich toleranter für die "ganze Bild"-Erkennung
    if (areaNorm < 0.005)
        return SegmentationAttempt(null, "hq: Region zu klein area=${format3(areaNorm)}")

    // Skalpel-scharfe Ecken via Quadrant-Score (konvexe Huelle naehern)
    var tlX = region.minX; var tlY = region.minY
    var trX = region.maxX; var trY = region.minY
    var brX = region.maxX; var brY = region.maxY
    var blX = region.minX; var blY = region.maxY
    var tlS = Int.MAX_VALUE; var trS = Int.MIN_VALUE
    var brS = Int.MIN_VALUE; var blS = Int.MAX_VALUE

    for (y in region.minY..region.maxY) {
        for (x in region.minX..region.maxX) {
            if (!processed[y * sw + x]) continue
            val s1 = x + y; val s2 = x - y
            if (s1 < tlS) { tlS = s1; tlX = x; tlY = y }
            if (s2 > trS) { trS = s2; trX = x; trY = y }
            if (s1 > brS) { brS = s1; brX = x; brY = y }
            if (s2 < blS) { blS = s2; blX = x; blY = y }
        }
    }

    val conf = (fill * 0.5 + (1.0 - kotlin.math.abs(areaNorm - 0.30)) * 0.5).coerceIn(0.0, 1.0)
    return SegmentationAttempt(
        AutoBoardResult(
            tlX = (tlX.toDouble() / sw).coerceIn(0.0, 1.0),
            tlY = (tlY.toDouble() / sh).coerceIn(0.0, 1.0),
            trX = (trX.toDouble() / sw).coerceIn(0.0, 1.0),
            trY = (trY.toDouble() / sh).coerceIn(0.0, 1.0),
            brX = (brX.toDouble() / sw).coerceIn(0.0, 1.0),
            brY = (brY.toDouble() / sh).coerceIn(0.0, 1.0),
            blX = (blX.toDouble() / sw).coerceIn(0.0, 1.0),
            blY = (blY.toDouble() / sh).coerceIn(0.0, 1.0),
            confidence = conf, segmentedFill = fill, segmentedAreaNorm = areaNorm
        ),
        "hq:otsu($thresh) area=${format3(areaNorm)} fill=${format3(fill)}"
    )
}

/**
 * Erkennt gedruckte Targets (dunkle Punkte/Kreise) auf einem hellen Board.
 * Gibt normalisierte Koordinaten sortiert nach Naehe zur Board-Mitte zurueck.
 */
private fun detectTargetsOnBoard(
    frame: FrameData,
    tlX: Double, tlY: Double, trX: Double, trY: Double,
    brX: Double, brY: Double, blX: Double, blY: Double
): List<Pair<Double, Double>> {
    val rgba = frame.rgba ?: return emptyList()
    val fw = frame.width; val fh = frame.height

    val maxDim = 640
    val step = maxOf(1, maxOf(fw, fh) / maxDim)
    val sw = (fw / step).coerceAtLeast(8)
    val sh = (fh / step).coerceAtLeast(8)

    val bx0 = (minOf(tlX, blX) * sw).toInt().coerceIn(0, sw - 1)
    val bx1 = (maxOf(trX, brX) * sw).toInt().coerceIn(0, sw - 1)
    val by0 = (minOf(tlY, trY) * sh).toInt().coerceIn(0, sh - 1)
    val by1 = (maxOf(blY, brY) * sh).toInt().coerceIn(0, sh - 1)
    if (bx1 - bx0 < 4 || by1 - by0 < 4) return emptyList()

    // Mittlere Helligkeit des Boards
    var brightSum = 0L; var cnt = 0L
    for (y in by0..by1) for (x in bx0..bx1) {
        val srcY = minOf(fh - 1, y * step); val srcX = minOf(fw - 1, x * step)
        val pi = (srcY * fw + srcX) * 4
        val r = (rgba.getOrNull(pi)?.toInt() ?: 0) and 0xFF
        val g = (rgba.getOrNull(pi + 1)?.toInt() ?: 0) and 0xFF
        val b = (rgba.getOrNull(pi + 2)?.toInt() ?: 0) and 0xFF
        brightSum += maxOf(r, g, b); cnt++
    }
    if (cnt == 0L) return emptyList()
    val bgBright = (brightSum / cnt).toInt()
    if (bgBright < 80) return emptyList()

    val darkThresh = (bgBright * 0.55).toInt()
    val darkMask = BooleanArray(sw * sh)
    for (y in by0..by1) for (x in bx0..bx1) {
        val srcY = minOf(fh - 1, y * step); val srcX = minOf(fw - 1, x * step)
        val pi = (srcY * fw + srcX) * 4
        val r = (rgba.getOrNull(pi)?.toInt() ?: 255) and 0xFF
        val g = (rgba.getOrNull(pi + 1)?.toInt() ?: 255) and 0xFF
        val bv = (rgba.getOrNull(pi + 2)?.toInt() ?: 255) and 0xFF
        darkMask[y * sw + x] = maxOf(r, g, bv) < darkThresh
    }

    val minBlobPx = maxOf(2, (sw * sh) / 8000)
    val maxBlobPx = (sw * sh) / 30
    val visited = BooleanArray(sw * sh)
    val queue = IntArray(sw * sh)
    val targets = mutableListOf<Pair<Double, Double>>()
    val dx4 = intArrayOf(0, 0, -1, 1); val dy4 = intArrayOf(-1, 1, 0, 0)

    for (start in darkMask.indices) {
        if (!darkMask[start] || visited[start]) continue
        val sx = start % sw; val sy = start / sw
        if (sx < bx0 || sx > bx1 || sy < by0 || sy > by1) { visited[start] = true; continue }
        var head = 0; var tail = 0
        queue[tail++] = start; visited[start] = true
        var sumX = 0L; var sumY = 0L; var area = 0
        while (head < tail) {
            val idx = queue[head++]; area++
            val x = idx % sw; val y = idx / sw
            sumX += x; sumY += y
            for (d in 0..3) {
                val nx = x + dx4[d]; val ny = y + dy4[d]
                if (nx < bx0 || nx > bx1 || ny < by0 || ny > by1) continue
                val nidx = ny * sw + nx
                if (darkMask[nidx] && !visited[nidx]) { visited[nidx] = true; queue[tail++] = nidx }
            }
        }
        if (area in minBlobPx..maxBlobPx) {
            targets.add((sumX.toDouble() / area / sw) to (sumY.toDouble() / area / sh))
        }
    }
    val cx = (bx0 + bx1) / 2.0 / sw; val cy = (by0 + by1) / 2.0 / sh
    return targets.sortedBy { (x, y) -> (x - cx) * (x - cx) + (y - cy) * (y - cy) }
}

private data class RuntimeMemoryProfile(
    val lowMemoryMode: Boolean,
    val ringMaxBytes: Long,
    val ringMaxFrames: Int
)

private fun resolveRuntimeMemoryProfile(context: Context): RuntimeMemoryProfile {
    val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
    val memoryClassMb = am?.memoryClass ?: 192
    return when {
        memoryClassMb <= 192 -> RuntimeMemoryProfile(lowMemoryMode = true, ringMaxBytes = 22L * 1024L * 1024L, ringMaxFrames = 120)
        memoryClassMb <= 256 -> RuntimeMemoryProfile(lowMemoryMode = false, ringMaxBytes = 34L * 1024L * 1024L, ringMaxFrames = 150)
        else -> RuntimeMemoryProfile(lowMemoryMode = false, ringMaxBytes = 46L * 1024L * 1024L, ringMaxFrames = 180)
    }
}

private fun decodeSampledBitmapFromUri(context: Context, uri: String): Bitmap? {
    if (uri.isBlank()) return null
    val reqSizePx = 256
    return try {
        val imageUri = uri.toUri()
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(imageUri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
        }
        val srcW = bounds.outWidth.coerceAtLeast(1)
        val srcH = bounds.outHeight.coerceAtLeast(1)
        var sample = 1
        while ((srcW / sample) > reqSizePx * 2 || (srcH / sample) > reqSizePx * 2) {
            sample *= 2
        }
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample.coerceAtLeast(1)
            inPreferredConfig = Bitmap.Config.RGB_565
        }
        context.contentResolver.openInputStream(imageUri)?.use {
            BitmapFactory.decodeStream(it, null, opts)
        }
    } catch (_: Throwable) {
        null
    }
}

private enum class GameMode {
    MOST_POINTS,
    FASTEST_CLEAR,
    PISTOL_ONLY,
    DRILL
}

private enum class OnboardingStep {
    PROFILE,
    PERMISSIONS,
    BOARD,
    CALIBRATION,
    READY
}

private enum class AppTab(val label: String) {
    LOSLEGEN("Loslegen"),
    SPIELER("Spieler"),
    WAFFEN("Waffen"),
    SETUP("Setup"),
    SYSTEM("System"),
    DETAILS("Details")
}

private fun onboardingStepInstruction(step: OnboardingStep): String = when (step) {
    OnboardingStep.PROFILE -> "Trage Spielername und Waffenname ein, dann 'Profil speichern'."
    OnboardingStep.PERMISSIONS -> "Erlaube Mikrofon und Kamera in der Abfrage."
    OnboardingStep.BOARD -> "Starte Segmentierung der Zielplatte. Falls noetig danach 4 Ecken manuell setzen und feinziehen."
    OnboardingStep.CALIBRATION -> "Starte Kalibrierung, danach 5 saubere Schuesse. Die Zaehlung laeuft automatisch."
    OnboardingStep.READY -> "Alles erledigt. Jetzt Start druecken."
}

private fun gameModeLabel(mode: GameMode): String = when (mode) {
    GameMode.MOST_POINTS -> "Meiste Punkte"
    GameMode.FASTEST_CLEAR -> "Schnellste Ziele"
    GameMode.PISTOL_ONLY -> "Pistol Only"
    GameMode.DRILL -> "Drill"
}

private fun isPistolWeapon(name: String): Boolean {
    val n = name.lowercase(Locale.ROOT)
    return n.contains("pistol") || n.contains("pistole") || n.contains("glock")
}

private fun zoneName(xNorm: Double, yNorm: Double): String {
    val xLeft = xNorm < 0.5
    val yTop = yNorm < 0.5
    return when {
        yTop && xLeft -> "oben links"
        yTop && !xLeft -> "oben rechts"
        !yTop && xLeft -> "unten links"
        else -> "unten rechts"
    }
}

private fun nextDrillCommand(random: Random): Pair<String, Int> {
    val zones = listOf("oben links", "oben rechts", "unten links", "unten rechts")
    return zones[random.nextInt(zones.size)] to (1 + random.nextInt(2))
}

private fun pointInPolygon(
    x: Double,
    y: Double,
    polygon: List<Pair<Double, Double>>
): Boolean {
    if (polygon.size < 3) return false
    var inside = false
    var j = polygon.lastIndex
    for (i in polygon.indices) {
        val xi = polygon[i].first
        val yi = polygon[i].second
        val xj = polygon[j].first
        val yj = polygon[j].second
        val dy = yj - yi
        val intersects = if ((yi > y) != (yj > y) && kotlin.math.abs(dy) > 1e-9) {
            x < (xj - xi) * (y - yi) / dy + xi
        } else {
            false
        }
        if (intersects) inside = !inside
        j = i
    }
    return inside
}

/**
 * ViewModel: haelt spielrelevanten State ueber Screenrotation hinaus.
 */
class AirsoftViewModel : ViewModel() {
    var scoreP1: Int = 0
    var scoreP2: Int = 0
    var hitsP1: Int = 0
    var hitsP2: Int = 0
    var fastestP1Ms: Long? = null
    var fastestP2Ms: Long? = null
    var roundStartMs: Long = 0L
    var highScore: Int = 0
    var highScoreOwner: String = "-"
    var highScoreWeapon: String = "-"
    var calibrationShots: Int = 0
    var calibrationPeakSum: Double = 0.0
    var lastReplayPath: String? = null
    internal val shotHistory = mutableListOf<ShotRecord>()
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            EdgertonAirsoftTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    AirsoftPipelineScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@Composable
private fun AirsoftPipelineScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainExecutor = remember(context) { ContextCompat.getMainExecutor(context) }
    val runtimeMemoryProfile = remember(context) { resolveRuntimeMemoryProfile(context) }

    // Tactical Colors - Unified Palette
    val tactDark = Color(0xFF0A0A0F)      // Deep Black-Blue
    val tactGray = Color(0xFF14141A)      // Surface
    val tactGrayLight = Color(0xFF22222B) // Lighter Surface
    val tactAccent = Color(0xFF00E676)    // Neon Green (Active)
    val tactAmber = Color(0xFFFFB300)     // Warning/Player 2
    val tactRed = Color(0xFFFF3D00)       // Alert/Miss
    val tactText = Color(0xFFECEFF1)      // Main Text
    val tactTextDim = Color(0xFF78909C)   // Secondary Text
    val tactBorder = Color(0xFF2D2D3A)    // Borders


    val vm: AirsoftViewModel = viewModel()
    // Segmentation preference: "AUTO", "SAM", "SAM3"
    var segPreference by remember { mutableStateOf("AUTO") }

    var configPeakThresh by remember { mutableFloatStateOf(3.5f) }
    var configCooldownMs by remember { mutableFloatStateOf(350f) }
    var configDiffThreshold by remember { mutableIntStateOf(30) }

    var replayVideoPath by remember { mutableStateOf(vm.lastReplayPath) }
    var showReplayViewer by remember { mutableStateOf(false) }

    val detector = remember { ShotDetectionEngine() }
    val audioRingBuffer = remember { AudioRingBuffer(retentionMs = 3_000) }
    val ringBuffer = remember(runtimeMemoryProfile) {
        FrameRingBuffer(
            retentionMs = 2_000,
            maxBytes = runtimeMemoryProfile.ringMaxBytes,
            maxFrames = runtimeMemoryProfile.ringMaxFrames
        )
    }
    val motionAnalyzer = remember { MotionAnalyzer() }
    val triggerCoordinator = remember { TriggerCoordinator() }
    val exporter = remember {
        FallbackReplayExporter(
            primary = Mp4SlowMotionExporter(context),
            fallback = ImageSequenceExporter(context)
        )
    }
    val mlSegmenter = remember(context) { HybridSegmenter(context) }
    val cameraController = remember { CameraSessionController(context) }
    val trajectoryAnalyzer = remember { TrajectoryAnalyzer() }
    val latestMotionScore = remember { AtomicReference(0.0) }
    val prefs = remember(context) { context.getSharedPreferences("airsoft_profiles", Context.MODE_PRIVATE) }
    val random = remember { Random(SystemClock.elapsedRealtime()) }
    val uiScope = rememberCoroutineScope()

    var previewView by remember { mutableStateOf<PreviewView?>(null) }
    var detectorState by remember { mutableStateOf(DetectorState()) }
    var motionState by remember { mutableStateOf(MotionState()) }
    var overlayState by remember { mutableStateOf(OverlayViewState()) }
    var shotCount by remember { mutableIntStateOf(0) }
    var triggerCount by remember { mutableIntStateOf(0) }
    var ringSize by remember { mutableIntStateOf(0) }
    var audioRingSize by remember { mutableIntStateOf(0) }
    var frameTick by remember { mutableIntStateOf(0) }
    var audioTick by remember { mutableIntStateOf(0) }
    var lastShotAt by remember { mutableStateOf("-") }
    var lastAudioAt by remember { mutableStateOf("-") }
    var lastExport by remember { mutableStateOf("-") }
    var lastAudioMux by remember { mutableStateOf("-") }
    var lastTrajectory by remember { mutableStateOf("-") }
    var lastHit by remember { mutableStateOf("-") }
    var lastDetectorSource by remember { mutableStateOf("diff") }
    var lastYolo by remember { mutableStateOf("0.000") }
    var playerName by remember { mutableStateOf("Player1") }
    var player2Name by remember { mutableStateOf("John Doe") }
    var loadoutName by remember { mutableStateOf("Standard") }
    var playerCount by remember { mutableIntStateOf(1) }
    var weaponName by remember { mutableStateOf("Replica A") }
    var weaponPhotoUri by remember { mutableStateOf("") }
    var weaponFpsInput by remember { mutableStateOf("") }
    var weaponBbWeightInput by remember { mutableStateOf("0.20") }
    var weaponHopUpInput by remember { mutableStateOf("0") }
    var weaponNotesInput by remember { mutableStateOf("") }
    var weaponProfiles by remember { mutableStateOf<List<WeaponProfile>>(emptyList()) }
    var selectedWeaponId by remember { mutableStateOf("") }
    var player1WeaponId by remember { mutableStateOf("") }
    var player2WeaponId by remember { mutableStateOf("") }
    var gameMode by remember { mutableStateOf(GameMode.MOST_POINTS) }
    var activePlayer by remember { mutableIntStateOf(1) }
    var hasMicPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        )
    }
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    var activeTab by remember { mutableStateOf(AppTab.LOSLEGEN) }
    var onboardingDone by remember { mutableStateOf(false) }
    var onboardingStep by remember { mutableStateOf(OnboardingStep.PROFILE) }
    var setupLayerDismissed by remember { mutableStateOf(false) }
    var calibrationLastAcceptedMs by remember { mutableLongStateOf(0L) }
    var calibrationStartedAtMs by remember { mutableLongStateOf(0L) }
    var calibrationIgnoredCount by remember { mutableIntStateOf(0) }
    var hitMarkerToken by remember { mutableLongStateOf(0L) }
    var activeDraggedCorner by remember { mutableStateOf<Int?>(null) }
    val latestFrameForAutoDetect = remember { AtomicReference<FrameData?>(null) }
    var lastAutoDetect by remember { mutableStateOf("-") }
    var lastSegmentationBackend by remember { mutableStateOf("heuristic") }
    var isAutoDetectRunning by remember { mutableStateOf(false) }
    var autoDetectProgress by remember { mutableFloatStateOf(0f) }
    var autoDetectPhase by remember { mutableStateOf("") }
    var autoDetectStateLabel by remember { mutableStateOf("idle") }
    var autoDetectRuntimeMs by remember { mutableLongStateOf(0L) }
    var autoDetectErrorCode by remember { mutableStateOf("-") }

    // ── Zweistufiger Segmentierungs-Picker ──────────────────────────────
    // Schritt 1: Foto aufnehmen → Picker anzeigen
    // Schritt 2: Tap auf Foto → SAM segmentiert exakt den getippten Punkt
    val capturedSegFrameRef = remember { AtomicReference<FrameData?>(null) }
    var capturedSegBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var showSegmentationPicker by remember { mutableStateOf(false) }
    var samScanResults by remember { mutableStateOf<List<Bitmap>>(emptyList()) }
    var isSamScanRunning by remember { mutableStateOf(false) }
    var autoScanRequested by remember { mutableStateOf(false) }
    // Multi-Tap-Stage-2: User tippt einzelne Sub-Ziele auf der bereits
    // segmentierten Platte an. Jeder Tap ruft EfficientSAM erneut auf, aber
    // restriktiv auf die Stage-1-Maske.
    var subTargetMode by remember { mutableStateOf(false) }
    var nextSubTargetId by remember { mutableIntStateOf(1) }
    var isSubTargetSegmenting by remember { mutableStateOf(false) }
    // Kamera-Vorschau ist im Spielbetrieb verschwendete Bildschirmflaeche.
    // Default: aus. Sichtbar waehrend Setup, Auto-Erkennung, oder wenn aktiv.
    var cameraExpanded by remember { mutableStateOf(false) }
    // Status-String fuer das Segmentier-Backend in der UI.
    var samDiagText by remember { mutableStateOf("Segmentierung: OpenCV-FloodFill (lokal, bereit)") }

    var scoreP1 by remember { mutableIntStateOf(vm.scoreP1) }
    var scoreP2 by remember { mutableIntStateOf(vm.scoreP2) }
    var hitsP1 by remember { mutableIntStateOf(vm.hitsP1) }
    var hitsP2 by remember { mutableIntStateOf(vm.hitsP2) }
    var highScore by remember { mutableIntStateOf(vm.highScore) }
    var highScoreOwner by remember { mutableStateOf(vm.highScoreOwner) }
    var highScoreWeapon by remember { mutableStateOf(vm.highScoreWeapon) }
    var fastestP1Ms by remember { mutableStateOf(vm.fastestP1Ms) }
    var fastestP2Ms by remember { mutableStateOf(vm.fastestP2Ms) }
    var roundStartMs by remember { mutableLongStateOf(vm.roundStartMs) }
    var calibrationShots by remember { mutableIntStateOf(vm.calibrationShots) }
    var calibrationPeakSum by remember { mutableDoubleStateOf(vm.calibrationPeakSum) }
    var shotHistory by remember { mutableStateOf(vm.shotHistory.toList()) }

    var hitValidationInfo by remember { mutableStateOf("-") }
    var targetXNorm by remember { mutableDoubleStateOf(0.5) }
    var targetYNorm by remember { mutableDoubleStateOf(0.5) }
    var boardTlXNorm by remember { mutableStateOf<Double?>(null) }
    var boardTlYNorm by remember { mutableStateOf<Double?>(null) }
    var boardTrXNorm by remember { mutableStateOf<Double?>(null) }
    var boardTrYNorm by remember { mutableStateOf<Double?>(null) }
    var boardBrXNorm by remember { mutableStateOf<Double?>(null) }
    var boardBrYNorm by remember { mutableStateOf<Double?>(null) }
    var boardBlXNorm by remember { mutableStateOf<Double?>(null) }
    var boardBlYNorm by remember { mutableStateOf<Double?>(null) }
    var drillZone by remember { mutableStateOf("oben links") }
    var drillRemaining by remember { mutableIntStateOf(1) }
    var drillAwaitingWeaponSwitch by remember { mutableStateOf(false) }
    var drillStepsDone by remember { mutableIntStateOf(0) }
    var calibrationInfo by remember { mutableStateOf("Noch nicht kalibriert") }
    var isFiveShotCalibrationActive by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("Bereit") }
    var isRunning by remember { mutableStateOf(false) }
    var pendingStartAfterPermission by remember { mutableStateOf(false) }
    var firstFrameSeen by remember { mutableStateOf(false) }
    var firstAudioSeen by remember { mutableStateOf(false) }

    LaunchedEffect(scoreP1, scoreP2, hitsP1, hitsP2, highScore,
        highScoreOwner, highScoreWeapon, fastestP1Ms, fastestP2Ms,
        roundStartMs, calibrationShots, calibrationPeakSum) {
        vm.scoreP1 = scoreP1; vm.scoreP2 = scoreP2
        vm.hitsP1 = hitsP1; vm.hitsP2 = hitsP2
        vm.highScore = highScore; vm.highScoreOwner = highScoreOwner
        vm.highScoreWeapon = highScoreWeapon
        vm.fastestP1Ms = fastestP1Ms; vm.fastestP2Ms = fastestP2Ms
        vm.roundStartMs = roundStartMs
        vm.calibrationShots = calibrationShots
        vm.calibrationPeakSum = calibrationPeakSum
    }

    LaunchedEffect(configPeakThresh, configCooldownMs) {
        detector.updateConfig(
            detector.config.copy(
                peakToNoiseThreshold = configPeakThresh.toDouble(),
                cooldownMs = configCooldownMs.toLong()
            )
        )
    }
    LaunchedEffect(configDiffThreshold) {
        trajectoryAnalyzer.diffThreshold = configDiffThreshold
    }

    var ttsReady by remember { mutableStateOf(false) }
    val tts = remember(context) {
        TextToSpeech(context) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
        }
    }

    fun speak(text: String) {
        if (!ttsReady) return
        tts.language = Locale.GERMAN
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "airsoft_tts")
    }

    fun weaponNameForId(id: String): String {
        return weaponProfiles.firstOrNull { it.id == id }?.name ?: "Nicht zugewiesen"
    }

    fun applyWeaponProfileById(id: String) {
        if (id.isBlank()) return
        val profile = weaponProfiles.firstOrNull { it.id == id } ?: return
        selectedWeaponId = profile.id
        weaponName = profile.name
        weaponPhotoUri = profile.photoUri
        weaponFpsInput = if (profile.fps <= 0) "" else profile.fps.toString()
        weaponBbWeightInput = format3(profile.bbWeightG)
        weaponHopUpInput = profile.hopUp.toString()
        weaponNotesInput = profile.notes
    }

    fun ensureWeaponAssignments() {
        if (weaponProfiles.isEmpty()) return
        val defaultId = selectedWeaponId.ifBlank { weaponProfiles.first().id }
        if (player1WeaponId.isBlank()) {
            player1WeaponId = defaultId
        }
        if (player2WeaponId.isBlank()) {
            player2WeaponId = weaponProfiles.firstOrNull { it.id != player1WeaponId }?.id ?: player1WeaponId
        }
        if (playerCount <= 1 && activePlayer != 1) {
            activePlayer = 1
        }
    }

    fun assignSelectedWeaponToPlayer(player: Int) {
        val id = selectedWeaponId.ifBlank { "w_${SystemClock.elapsedRealtime()}" }
        val profile = WeaponProfile(
            id = id,
            name = weaponName.ifBlank { "Waffe" },
            photoUri = weaponPhotoUri,
            fps = weaponFpsInput.toIntOrNull() ?: 0,
            bbWeightG = weaponBbWeightInput.replace(',', '.').toDoubleOrNull() ?: 0.20,
            hopUp = weaponHopUpInput.toIntOrNull() ?: 0,
            notes = weaponNotesInput
        )
        val updatedProfiles = weaponProfiles.toMutableList()
        val idx = updatedProfiles.indexOfFirst { it.id == id }
        if (idx >= 0) updatedProfiles[idx] = profile else updatedProfiles.add(profile)
        weaponProfiles = updatedProfiles
        selectedWeaponId = id
        ensureWeaponAssignments()
        val assignedId = selectedWeaponId
        if (player == 1) {
            player1WeaponId = assignedId
        } else {
            player2WeaponId = assignedId
        }
        if (activePlayer == player || (player == 1 && playerCount <= 1)) {
            applyWeaponProfileById(assignedId)
        }
        statusText = "${if (player == 1) playerName.ifBlank { "Spieler 1" } else player2Name.ifBlank { "John Doe" }} nutzt jetzt ${weaponNameForId(assignedId)}"
    }

    fun applyAssignedWeaponForPlayer(player: Int) {
        ensureWeaponAssignments()
        val assignedId = if (player == 1) player1WeaponId else player2WeaponId
        if (assignedId.isNotBlank()) {
            applyWeaponProfileById(assignedId)
        }
    }

    fun setActivePlayerSlot(player: Int) {
        val nextPlayer = if (playerCount <= 1) 1 else player.coerceIn(1, 2)
        activePlayer = nextPlayer
        applyAssignedWeaponForPlayer(nextPlayer)
        statusText = "Aktiver Spieler: ${if (nextPlayer == 1) playerName.ifBlank { "Spieler 1" } else player2Name.ifBlank { "John Doe" }}"
    }

    fun upsertCurrentWeaponProfile() {
        val id = selectedWeaponId.ifBlank { "w_${SystemClock.elapsedRealtime()}" }
        val fps = weaponFpsInput.toIntOrNull() ?: 0
        val bb = weaponBbWeightInput.replace(',', '.').toDoubleOrNull() ?: 0.20
        val hop = weaponHopUpInput.toIntOrNull() ?: 0
        val profile = WeaponProfile(
            id = id,
            name = weaponName.ifBlank { "Waffe" },
            photoUri = weaponPhotoUri,
            fps = fps,
            bbWeightG = bb,
            hopUp = hop,
            notes = weaponNotesInput
        )
        val updated = weaponProfiles.toMutableList()
        val idx = updated.indexOfFirst { it.id == id }
        if (idx >= 0) {
            updated[idx] = profile
        } else {
            updated.add(profile)
        }
        weaponProfiles = updated
        selectedWeaponId = id
    }

    fun applySelectedWeaponProfile() {
        val profile = weaponProfiles.firstOrNull { it.id == selectedWeaponId } ?: return
        weaponName = profile.name
        weaponPhotoUri = profile.photoUri
        weaponFpsInput = if (profile.fps <= 0) "" else profile.fps.toString()
        weaponBbWeightInput = format3(profile.bbWeightG)
        weaponHopUpInput = profile.hopUp.toString()
        weaponNotesInput = profile.notes
    }

    fun saveProfiles() {
        upsertCurrentWeaponProfile()
        ensureWeaponAssignments()
        prefs.edit {
            putString("player_name_1", playerName)
            putString("player_name_2", player2Name)
            putString("loadout_name", loadoutName)
            putInt("player_count", playerCount)
            putString("weapon_name", weaponName)
            putString("weapon_photo_uri", weaponPhotoUri)
            putString("weapon_profiles_json", weaponProfilesToJson(weaponProfiles))
            putString("selected_weapon_id", selectedWeaponId)
            putString("player1_weapon_id", player1WeaponId)
            putString("player2_weapon_id", player2WeaponId)
            putInt("active_player_slot", activePlayer)
            putInt("high_score", highScore)
            putString("high_score_owner", highScoreOwner)
            putString("high_score_weapon", highScoreWeapon)
            putBoolean("onboarding_done", onboardingDone)
        }
    }

    fun updateOnboardingStep() {
        if (onboardingDone) {
            onboardingStep = OnboardingStep.READY
            return
        }
        val profileReady = playerName.isNotBlank() && weaponName.isNotBlank()
        val permissionReady = hasMicPermission && hasCameraPermission
        // SAM-Maske ODER alle 4 Ecken gesetzt
        val boardReady = overlayState.maskBitmap != null ||
            (boardTlXNorm != null && boardTrXNorm != null && boardBrXNorm != null && boardBlXNorm != null)
        val calibrationReady = calibrationShots >= 5
        onboardingStep = when {
            !profileReady -> OnboardingStep.PROFILE
            !permissionReady -> OnboardingStep.PERMISSIONS
            !boardReady -> OnboardingStep.BOARD
            !calibrationReady -> OnboardingStep.CALIBRATION
            else -> OnboardingStep.READY
        }
    }

    fun focusUiForOnboardingStep(step: OnboardingStep) {
        if (onboardingDone) return
        when (step) {
            OnboardingStep.PROFILE     -> activeTab = AppTab.SPIELER
            OnboardingStep.PERMISSIONS -> activeTab = AppTab.SETUP
            OnboardingStep.BOARD,
            OnboardingStep.CALIBRATION -> activeTab = AppTab.SETUP
            else -> {}
        }
    }

    LaunchedEffect(Unit) {
        playerName = prefs.getString("player_name_1", playerName) ?: playerName
        player2Name = prefs.getString("player_name_2", player2Name) ?: player2Name
        loadoutName = prefs.getString("loadout_name", loadoutName) ?: loadoutName
        playerCount = prefs.getInt("player_count", playerCount).coerceIn(1, 2)
        weaponName = prefs.getString("weapon_name", weaponName) ?: weaponName
        weaponPhotoUri = prefs.getString("weapon_photo_uri", "") ?: ""
        weaponProfiles = weaponProfilesFromJson(prefs.getString("weapon_profiles_json", null))
        selectedWeaponId = prefs.getString("selected_weapon_id", "") ?: ""
        player1WeaponId = prefs.getString("player1_weapon_id", "") ?: ""
        player2WeaponId = prefs.getString("player2_weapon_id", "") ?: ""
        activePlayer = prefs.getInt("active_player_slot", activePlayer).coerceIn(1, 2)
        if (weaponProfiles.isEmpty()) {
            selectedWeaponId = "w_default"
            weaponProfiles = listOf(
                WeaponProfile(
                    id = selectedWeaponId,
                    name = weaponName,
                    photoUri = weaponPhotoUri,
                    fps = 0,
                    bbWeightG = 0.20,
                    hopUp = 0,
                    notes = ""
                )
            )
        }
        if (selectedWeaponId.isBlank()) {
            selectedWeaponId = weaponProfiles.first().id
        }
        ensureWeaponAssignments()
        applyAssignedWeaponForPlayer(activePlayer)
        highScore = prefs.getInt("high_score", 0)
        highScoreOwner = prefs.getString("high_score_owner", "-") ?: "-"
        highScoreWeapon = prefs.getString("high_score_weapon", "-") ?: "-"
        onboardingDone = prefs.getBoolean("onboarding_done", false)
        setupLayerDismissed = onboardingDone
        activeTab = if (onboardingDone) AppTab.LOSLEGEN else AppTab.SPIELER
        updateOnboardingStep()
        focusUiForOnboardingStep(onboardingStep)
        if (!onboardingDone) {
            statusText = "Erststart: folge dem roten Pfad"
        }
        val (zone, reps) = nextDrillCommand(random)
        drillZone = zone
        drillRemaining = reps
    }

    // Status-Banner: zeigt den aktiven Segmentier-Backend an. SAM wurde aus dem
    // heissen Pfad genommen (zuverlaessig lokal nicht ans Laufen gebracht); statt-
    // dessen laeuft OpenCV-FloodFill (gleich wie QmStone-Referenz).
    LaunchedEffect(lastSegmentationBackend) {
        samDiagText = when {
            lastSegmentationBackend.startsWith("floodfill") -> "Segmentierung: OpenCV-FloodFill (lokal, aktiv)"
            lastSegmentationBackend == "failed" -> "Segmentierung: Fehler beim letzten Versuch"
            lastSegmentationBackend.startsWith("hq") -> "Segmentierung: Heuristik-Box (FloodFill lieferte nichts)"
            lastSegmentationBackend.isBlank() -> "Segmentierung: OpenCV-FloodFill (lokal, bereit)"
            else -> "Segmentierung: $lastSegmentationBackend"
        }
    }

    LaunchedEffect(onboardingStep, onboardingDone) {
        focusUiForOnboardingStep(onboardingStep)
    }

    // Frueher: Pre-init der SAM ONNX-Modelle. Aktuell laeuft die Segmentierung
    // ueber LocalSegmenter (OpenCV-FloodFill, lokal, kein Modell noetig).
    // SamOnnxHelper bleibt als Code-Stub fuer einen spaeteren optionalen Pfad.

    // Kamera-Frame-Analyse nur wenn Match läuft – kein CPU-/Akku-Verbrauch im Standby
    LaunchedEffect(Unit) {
        withContext(Dispatchers.Default) {
            statusText = "Lade ML-Modell (SAM)..."
            SamOnnxHelper.initEfficientSam(context)
            if (SamOnnxHelper.isEfficientSamAvailable()) {
                statusText = "ML-Modell bereit"
                lastSegmentationBackend = "sam-ready"
            } else {
                statusText = "ML-Modell konnte nicht geladen werden - nutze FloodFill"
            }
        }
    }

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            weaponPhotoUri = uri.toString()
            saveProfiles()
            statusText = "Waffenfoto gespeichert"
        }
    }

    fun startNewRound() {
        roundStartMs = SystemClock.elapsedRealtime()
        scoreP1 = 0; scoreP2 = 0; hitsP1 = 0; hitsP2 = 0
        fastestP1Ms = null; fastestP2Ms = null
        hitValidationInfo = "Neue Runde gestartet"
        vm.shotHistory.clear(); shotHistory = emptyList()
        if (gameMode == GameMode.DRILL) {
            val (zone, reps) = nextDrillCommand(random)
            drillZone = zone; drillRemaining = reps
            drillAwaitingWeaponSwitch = false; drillStepsDone = 0
            speak("Drill startet. $zone mal $reps")
        }
    }

    fun startPipeline() {
        if (isRunning) { statusText = "Pipeline laeuft bereits"; return }
        pendingStartAfterPermission = false
        firstFrameSeen = false; firstAudioSeen = false
        if (roundStartMs == 0L) startNewRound()
        statusText = "Starte Audio + Kamera..."
        isRunning = true
        detector.start(object : ShotDetectionEngine.Listener {
            override fun onStateUpdate(state: DetectorState) {
                mainExecutor.execute { detectorState = state }
            }
            override fun onAudioChunk(chunk: AudioChunk) {
                audioRingBuffer.add(chunk)
                mainExecutor.execute {
                    audioRingSize = audioRingBuffer.size()
                    audioTick += 1
                    lastAudioAt = "${SystemClock.elapsedRealtime() - chunk.timestampMs} ms her"
                    if (!firstAudioSeen) {
                        firstAudioSeen = true
                        statusText = if (firstFrameSeen) "Pipeline aktiv (Audio+Kamera)" else "Audio aktiv, warte auf Kamera"
                    }
                }
            }
            override fun onShotDetected(event: ShotEvent) {
                try { handleShotDetected(event) } catch (t: Throwable) {
                    android.util.Log.e("ShotCrash", "onShotDetected crashed: peak=${event.peakToNoise} crest=${event.crest} hp=${event.highPassRatio}", t)
                    mainExecutor.execute { statusText = "Shot-Handler-Fehler abgefangen: ${t.message ?: t::class.java.simpleName}" }
                }
            }
            private fun handleShotDetected(event: ShotEvent) {
                // Sicherheits-Gate: Solange das Setup (Board + Kalibrierung) nicht fertig ist,
                // ignorieren wir Schüsse für die Analyse/Export, außer wir sind gerade im
                // expliziten Kalibrierungs-Modus.
                if (!onboardingDone && !isFiveShotCalibrationActive) {
                    return
                }

                val trigger = triggerCoordinator.tryTrigger(
                    timestampMs = event.timestampMs,
                    motionScore = latestMotionScore.get(),
                    audioPeakToNoise = event.peakToNoise
                )
                mainExecutor.execute {
                    shotCount += 1
                    lastShotAt = "${SystemClock.elapsedRealtime() - event.timestampMs} ms her"
                    var justFinishedCalibration = false
                    if (isFiveShotCalibrationActive && calibrationShots < 5) {
                        if (calibrationStartedAtMs > 0L && (SystemClock.elapsedRealtime() - calibrationStartedAtMs) > 90_000) {
                            isFiveShotCalibrationActive = false
                            calibrationInfo = "Kalibrierung abgebrochen: Timeout, bitte neu starten"
                            statusText = "Kalibrierung Timeout - neu starten"
                            return@execute
                        }
                        val ratioGuard = event.peakToNoise >= 3.5
                        val crestGuard = event.crest >= 2.5
                        val hpGuard = event.highPassRatio >= 0.50
                        val timeGuard = event.timestampMs - calibrationLastAcceptedMs >= 800
                        if (ratioGuard && crestGuard && hpGuard && timeGuard) {
                            val newShots = calibrationShots + 1
                            val newPeakSum = calibrationPeakSum + event.peakToNoise
                            val avg = newPeakSum / newShots
                            calibrationShots = newShots; calibrationPeakSum = newPeakSum
                            calibrationLastAcceptedMs = event.timestampMs
                            calibrationInfo = "Kalibrierung $newShots/5 (avg Peak/Noise ${format3(avg)})"
                            statusText = "Kalibrierung: Schuss $newShots/5 erkannt"
                            if (newShots == 5) {
                                isFiveShotCalibrationActive = false; calibrationStartedAtMs = 0L
                                calibrationInfo = "Kalibrierung fertig fuer $weaponName: avg Peak/Noise ${format3(avg)}"
                                statusText = "Kalibrierung abgeschlossen (5/5) - $weaponName"
                                updateOnboardingStep(); justFinishedCalibration = true
                            }
                        } else {
                            calibrationIgnoredCount += 1
                            val reason = when {
                                !timeGuard -> "zu nah am letzten Schuss (${event.timestampMs - calibrationLastAcceptedMs}ms)"
                                !ratioGuard -> "zu leise (ratio ${format3(event.peakToNoise)}, min 3.5)"
                                !crestGuard -> "kein Impuls (crest ${format3(event.crest)}, min 2.5)"
                                else -> "zu tieffrequent (hp ${format3(event.highPassRatio)}, min 0.50)"
                            }
                            calibrationInfo = "Kalibrierung: $reason (${calibrationIgnoredCount}x ignoriert)"
                        }
                    }
                    if (!justFinishedCalibration) {
                        statusText = String.format(Locale.US, "Schuss erkannt (%s / %s, ratio %.2f, motion %.3f)",
                            playerName, weaponName, event.peakToNoise, latestMotionScore.get())
                    }
                    // triggerActive existiert nicht in OverlayViewState – kein copy()-Aufruf nötig
                }
                if (trigger != null) {
                    // FPS Boost einschalten sobald Trigger registriert wird
                    cameraController.boostFps(true)
                    uiScope.launch {
                        delay(2000) // 2 Sekunden Boost fuer Replay-Qualitaet
                        cameraController.boostFps(false)
                    }

                    // ── EDGERTON-METHODE ──────────────────────────────────────
                    // Harold Edgerton brauchte keine High-Speed-Kamera:
                    // Mikrofon erkennt den Knall → Blitz feuert → scharfes Bild.
                    //
                    // Hier: Audio-Trigger = Mikrofon, captureStillFrame = Blitz.
                    // Ring-Buffer liefert das "Vorher"-Bild (letzter Frame vor dem Schuss),
                    // Still-Capture liefert das scharfe "Nachher"-Bild (Einschlag sichtbar).
                    // Vorher vs. Nachher = Impact-Erkennung auf qualitativ hochwertigen Frames.

                    val window = ringBuffer.snapshotWindow(centerTimestampMs = trigger.timestampMs, preMs = 1_500, postMs = 500)
                    val audioWindow = audioRingBuffer.snapshotWindow(centerTimestampMs = trigger.timestampMs, preMs = 1_500, postMs = 500)

                    // Edgerton-Blitz: scharfes Standbild NACH dem Schuss
                    cameraController.captureStillFrame { edgertonFrame ->
                      try {
                        // Analyse-Fenster: Ring-Buffer + Edgerton-Frame
                        val analysisWindow = if (edgertonFrame != null) {
                            window + edgertonFrame
                        } else {
                            window
                        }

                        val points = trajectoryAnalyzer.analyze(analysisWindow)

                        // Impact-Erkennung: Edgerton-Methode (Vorher vs. Nachher)
                        val impactHole = if (edgertonFrame != null && window.isNotEmpty()) {
                            // Bestes "Vorher"-Frame: das letzte Frame BEVOR der Trigger kam
                            val preFrames = window.filter { it.timestampMs <= trigger.timestampMs }
                            if (preFrames.isNotEmpty()) {
                                // Erstelle ein synthetisches Analysefenster: Vorher + Nachher
                                val edgertonPair = listOf(
                                    preFrames.first(),       // Referenz (ruhig, vor dem Schuss)
                                    preFrames.last(),        // Knapp vor dem Schuss
                                    edgertonFrame,           // Edgerton-"Blitz" (scharf, nach dem Schuss)
                                    edgertonFrame            // Duplikat fuer detectImpactHole Algorithmus
                                )
                                trajectoryAnalyzer.detectImpactHole(edgertonPair)
                            } else {
                                trajectoryAnalyzer.detectImpactHole(analysisWindow)
                            }
                        } else {
                            trajectoryAnalyzer.detectImpactHole(analysisWindow)
                        }

                        val edgertonUsed = edgertonFrame != null

                        exporter.export(trigger, window, audioWindow) { result ->
                            mainExecutor.execute {
                              try {
                                val lastPoint = points.lastOrNull()
                                val boardPolygon = if (boardTlXNorm != null && boardTlYNorm != null && boardTrXNorm != null && boardTrYNorm != null && boardBrXNorm != null && boardBrYNorm != null && boardBlXNorm != null && boardBlYNorm != null) {
                                    listOf(boardTlXNorm!! to boardTlYNorm!!, boardTrXNorm!! to boardTrYNorm!!, boardBrXNorm!! to boardBrYNorm!!, boardBlXNorm!! to boardBlYNorm!!)
                                } else null
                                val rawCandidateX = impactHole?.xNorm ?: lastPoint?.xNorm
                                val rawCandidateY = impactHole?.yNorm ?: lastPoint?.yNorm
                                val rawQualityOk = when {
                                    impactHole != null -> impactHole.confidence >= 0.10
                                    lastPoint != null -> lastPoint.score >= 0.15
                                    else -> false
                                }
                                // Kein Fallback auf Board-Mitte: Ohne visuellen Nachweis kein Treffer!
                                val candidateX = rawCandidateX
                                val candidateY = rawCandidateY
                                val qualityOk = rawQualityOk
                                // Masken-basierte oder Polygon-basierte Treffererkennung
                                val selectedTargetMasks = overlayState.subTargets.mapNotNull { it.maskBitmap }
                                val segMask = overlayState.maskBitmap
                                val targetDefined = selectedTargetMasks.isNotEmpty() || segMask != null || boardPolygon != null
                                val inTarget = if (candidateX != null && candidateY != null) {
                                    when {
                                        selectedTargetMasks.isNotEmpty() -> {
                                            selectedTargetMasks.any { subMask ->
                                                val px = (candidateX * subMask.width).toInt().coerceIn(0, subMask.width - 1)
                                                val py = (candidateY * subMask.height).toInt().coerceIn(0, subMask.height - 1)
                                                ((subMask.getPixel(px, py) ushr 24) and 0xFF) > 16
                                            }
                                        }
                                        segMask != null -> {
                                            // SAM-Maske: Pixel-Test
                                            val px = (candidateX * segMask.width).toInt().coerceIn(0, segMask.width - 1)
                                            val py = (candidateY * segMask.height).toInt().coerceIn(0, segMask.height - 1)
                                            ((segMask.getPixel(px, py) ushr 24) and 0xFF) > 16
                                        }
                                        boardPolygon != null -> pointInPolygon(candidateX, candidateY, boardPolygon)
                                        else -> false
                                    }
                                } else false
                                var hitValid = qualityOk && inTarget
                                var reason = when {
                                    candidateX == null -> "kein visueller Treffer erkannt"
                                    !qualityOk -> "Trefferpunkt zu unsicher (Confidence zu niedrig)"
                                    !targetDefined -> "kein Ziel fuer Verifizierung markiert"
                                    !inTarget && selectedTargetMasks.isNotEmpty() -> "ausserhalb der gewaehlten Ziele"
                                    !inTarget -> "ausserhalb markierter Zielplatte"
                                    else -> "ok"
                                }
                                val activeName = if (activePlayer == 1) playerName else player2Name
                                val roundElapsedMs = if (roundStartMs == 0L) 0L else SystemClock.elapsedRealtime() - roundStartMs
                                var awarded = 0
                                if (hitValid && gameMode == GameMode.PISTOL_ONLY && !isPistolWeapon(weaponName)) {
                                    hitValid = false; reason = "Pistol-Only: Waffenname passt nicht"
                                }
                                if (hitValid && gameMode == GameMode.DRILL) {
                                    val zone = zoneName(candidateX!!, candidateY!!)
                                    if (drillAwaitingWeaponSwitch) { hitValid = false; reason = "erst Waffenwechsel bestaetigen" }
                                    else if (zone != drillZone) { hitValid = false; reason = "falsche Drill-Zone ($zone statt $drillZone)" }
                                    else {
                                        awarded = 10; drillRemaining -= 1
                                        if (drillRemaining <= 0) {
                                            drillStepsDone += 1
                                            if (drillStepsDone % 3 == 0) { drillAwaitingWeaponSwitch = true; speak("Waffe wechseln und bestaetigen") }
                                            val (nextZone, reps) = nextDrillCommand(random)
                                            drillZone = nextZone; drillRemaining = reps
                                            speak("Naechstes Ziel $nextZone mal $reps")
                                        }
                                    }
                                } else if (hitValid) { awarded = 10 }
                                if (awarded > 0) {
                                    if (activePlayer == 1) {
                                        scoreP1 += awarded; hitsP1 += 1
                                        if (gameMode == GameMode.FASTEST_CLEAR && hitsP1 >= 10 && fastestP1Ms == null) fastestP1Ms = roundElapsedMs
                                    } else {
                                        scoreP2 += awarded; hitsP2 += 1
                                        if (gameMode == GameMode.FASTEST_CLEAR && hitsP2 >= 10 && fastestP2Ms == null) fastestP2Ms = roundElapsedMs
                                    }
                                    val activeScore = if (activePlayer == 1) scoreP1 else scoreP2
                                    if (activeScore > highScore) {
                                        highScore = activeScore; highScoreOwner = activeName; highScoreWeapon = weaponName; saveProfiles()
                                    }
                                }
                                triggerCount += 1
                                lastExport = if (result.ok) result.outputPath else "Fehler: ${result.error}"
                                if (result.ok && result.outputPath.isNotBlank()) { replayVideoPath = result.outputPath; vm.lastReplayPath = result.outputPath }
                                lastAudioMux = when {
                                    result.audioTrackMuxed -> "ja (${result.audioInfo ?: "AAC"})"
                                    result.audioInfo != null -> "nein (${result.audioInfo})"
                                    else -> "nein"
                                }
                                val edgertonHint = if (edgertonUsed) " [Edgerton]" else ""
                                lastTrajectory = if (points.isEmpty()) "keine Punkte$edgertonHint" else "${points.size} Punkte$edgertonHint"
                                lastHit = if (candidateX == null || candidateY == null) "kein Trefferpunkt" else String.format(Locale.US, "x=%.2f y=%.2f%s", candidateX, candidateY, edgertonHint)
                                hitValidationInfo = if (awarded > 0) "Treffer +$awarded fuer $activeName (${gameModeLabel(gameMode)})$edgertonHint" else "Kein Punkt: $reason"
                                val record = ShotRecord(playerName = activeName, timestampMs = trigger.timestampMs, xNorm = candidateX, yNorm = candidateY, awarded = awarded, reason = reason, roundElapsedMs = roundElapsedMs)
                                vm.shotHistory.add(0, record)
                                if (vm.shotHistory.size > 50) vm.shotHistory.removeAt(vm.shotHistory.lastIndex)
                                shotHistory = vm.shotHistory.toList()
                                statusText = if (result.ok) {
                                    val fallbackHint = if (result.fallbackUsed) " (Fallback)" else ""
                                    val audioHint = if (result.audioTrackMuxed) " + Audio" else ""
                                    "Trigger ${trigger.id}: ${result.frameCount} Frames als ${result.format}$audioHint$fallbackHint$edgertonHint"
                                } else "Export fehlgeschlagen: ${result.error}"
                                overlayState = overlayState.copy(hitXNorm = if (hitValid) candidateX else null, hitYNorm = if (hitValid) candidateY else null, hitWasValid = hitValid, targetXNorm = targetXNorm, targetYNorm = targetYNorm)
                                
                                // Sprachausgabe nur wenn Onboarding abgeschlossen und kein Setup-Modus aktiv
                                if (onboardingDone && !showSegmentationPicker && !isFiveShotCalibrationActive) {
                                    if (hitValid) { hitMarkerToken = SystemClock.elapsedRealtime(); speak("Treffer") }
                                    else if (targetDefined && candidateX != null) speak("Daneben")
                                }
                              } catch (t: Throwable) {
                                  android.util.Log.e("ShotCrash", "main-thread shot finalize crashed", t)
                                  statusText = "UI-Fehler nach Schuss: ${t.message ?: t::class.java.simpleName}"
                              }
                            }
                        }
                      } catch (t: Throwable) {
                          android.util.Log.e("ShotCrash", "edgerton capture callback crashed", t)
                          mainExecutor.execute { statusText = "Edgerton-Pfad-Fehler: ${t.message ?: t::class.java.simpleName}" }
                      }
                    }
                }
            }
            override fun onError(message: String) {
                mainExecutor.execute { statusText = message }
            }
        })
        statusText = "Pipeline laeuft (initialisiert)"
    }

    fun isBoardComplete(): Boolean {
        if (overlayState.maskBitmap != null) return true  // SAM-Maske = Board definiert
        return boardTlXNorm != null && boardTlYNorm != null && boardTrXNorm != null && boardTrYNorm != null &&
            boardBrXNorm != null && boardBrYNorm != null && boardBlXNorm != null && boardBlYNorm != null
    }

    fun updateTargetFromBoardCenter() {
        if (!isBoardComplete()) return
        val allX = listOfNotNull(boardTlXNorm, boardTrXNorm, boardBrXNorm, boardBlXNorm)
        val allY = listOfNotNull(boardTlYNorm, boardTrYNorm, boardBrYNorm, boardBlYNorm)
        targetXNorm = allX.average(); targetYNorm = allY.average()
    }

    fun updateFocusFromSelectedTargets() {
        val selectedTargets = overlayState.subTargets
        if (selectedTargets.isNotEmpty()) {
            targetXNorm = selectedTargets.map { it.centerXNorm }.average().coerceIn(0.0, 1.0)
            targetYNorm = selectedTargets.map { it.centerYNorm }.average().coerceIn(0.0, 1.0)
            overlayState = overlayState.copy(targetXNorm = targetXNorm, targetYNorm = targetYNorm)
            return
        }

        val mask = overlayState.maskBitmap
        if (mask != null) {
            val w = mask.width
            val h = mask.height
            val pixels = IntArray(w * h)
            mask.getPixels(pixels, 0, w, 0, 0, w, h)
            var sumX = 0L
            var sumY = 0L
            var count = 0L
            pixels.forEachIndexed { idx, px ->
                if (((px ushr 24) and 0xFF) > 16) {
                    sumX += idx % w
                    sumY += idx / w
                    count++
                }
            }
            if (count > 0L) {
                targetXNorm = (sumX.toDouble() / count / w).coerceIn(0.0, 1.0)
                targetYNorm = (sumY.toDouble() / count / h).coerceIn(0.0, 1.0)
                overlayState = overlayState.copy(targetXNorm = targetXNorm, targetYNorm = targetYNorm)
                return
            }
        }

        if (isBoardComplete()) {
            updateTargetFromBoardCenter()
            overlayState = overlayState.copy(targetXNorm = targetXNorm, targetYNorm = targetYNorm)
        }
    }

    fun setBoardCorner(index: Int, xNorm: Double, yNorm: Double) {
        when (index) {
            0 -> { boardTlXNorm = xNorm; boardTlYNorm = yNorm }
            1 -> { boardTrXNorm = xNorm; boardTrYNorm = yNorm }
            2 -> { boardBrXNorm = xNorm; boardBrYNorm = yNorm }
            3 -> { boardBlXNorm = xNorm; boardBlYNorm = yNorm }
        }
        updateTargetFromBoardCenter()
    }

    fun nearestBoardCornerIndex(xNorm: Double, yNorm: Double, maxDistanceNorm: Double = 0.10): Int? {
        val corners = listOf(boardTlXNorm to boardTlYNorm, boardTrXNorm to boardTrYNorm, boardBrXNorm to boardBrYNorm, boardBlXNorm to boardBlYNorm)
        var bestIndex: Int? = null; var bestDistSq = Double.MAX_VALUE
        corners.forEachIndexed { index, corner ->
            val cx = corner.first; val cy = corner.second
            if (cx != null && cy != null) {
                val dx = xNorm - cx; val dy = yNorm - cy; val distSq = dx * dx + dy * dy
                if (distSq < bestDistSq) { bestDistSq = distSq; bestIndex = index }
            }
        }
        return if (bestDistSq <= maxDistanceNorm * maxDistanceNorm) bestIndex else null
    }

    fun edgeCross(ax: Double, ay: Double, bx: Double, by: Double, cx: Double, cy: Double, dx: Double, dy: Double): Boolean {
        fun orient(px: Double, py: Double, qx: Double, qy: Double, rx: Double, ry: Double): Double = (qx - px) * (ry - py) - (qy - py) * (rx - px)
        val o1 = orient(ax, ay, bx, by, cx, cy); val o2 = orient(ax, ay, bx, by, dx, dy)
        val o3 = orient(cx, cy, dx, dy, ax, ay); val o4 = orient(cx, cy, dx, dy, bx, by)
        return (o1 * o2 < 0.0) && (o3 * o4 < 0.0)
    }

    fun isBoardGeometryValid(): Boolean {
        if (overlayState.maskBitmap != null) return true  // SAM-Maske braucht keine Geometrie-Validierung
        if (!isBoardComplete()) return false
        val tlx = boardTlXNorm!!; val tly = boardTlYNorm!!
        val trx = boardTrXNorm!!; val tryy = boardTrYNorm!!
        val brx = boardBrXNorm!!; val bry = boardBrYNorm!!
        val blx = boardBlXNorm!!; val bly = boardBlYNorm!!
        val area2 = kotlin.math.abs(tlx * tryy + trx * bry + brx * bly + blx * tly - (tly * trx + tryy * brx + bry * blx + bly * tlx))
        if (area2 < 0.02) return false
        if (edgeCross(tlx, tly, trx, tryy, brx, bry, blx, bly)) return false
        if (edgeCross(trx, tryy, brx, bry, blx, bly, tlx, tly)) return false
        return true
    }

    fun beginFiveShotCalibration() {
        if (!hasMicPermission) { statusText = "Mikrofonrecht fehlt - bitte Berechtigungen erlauben"; return }
        if (isFiveShotCalibrationActive && calibrationShots < 5) { statusText = "Kalibrierung laeuft bereits (${calibrationShots}/5)"; return }
        applyAssignedWeaponForPlayer(activePlayer)
        if (!isRunning) startPipeline()
        calibrationShots = 0; calibrationPeakSum = 0.0; calibrationLastAcceptedMs = 0L; calibrationIgnoredCount = 0
        calibrationStartedAtMs = SystemClock.elapsedRealtime()
        isFiveShotCalibrationActive = true
        calibrationInfo = "Kalibrierung gestartet: 5 verwertbare Schuesse werden automatisch gezaehlt"
        detector.startCalibration(durationMs = 90_000)
        statusText = "Kalibrierung aktiv (0/5) - jetzt 5x schiessen"
    }

    fun clearBoardCalibration() {
        boardTlXNorm = null; boardTlYNorm = null
        boardTrXNorm = null; boardTrYNorm = null
        boardBrXNorm = null; boardBrYNorm = null
        boardBlXNorm = null; boardBlYNorm = null
        overlayState = overlayState.copy(boardTlXNorm = null, boardTlYNorm = null, boardTrXNorm = null, boardTrYNorm = null, boardBrXNorm = null, boardBrYNorm = null, boardBlXNorm = null, boardBlYNorm = null, maskBitmap = null)
    }

    fun clearSamScanResults(keep: Bitmap? = null) {
        samScanResults.forEach { bmp ->
            if (bmp !== keep && !bmp.isRecycled) bmp.recycle()
        }
        samScanResults = emptyList()
        if (overlayState.samScanResults.isNotEmpty()) {
            overlayState = overlayState.copy(samScanResults = emptyList())
        }
    }

    suspend fun awaitCameraReady(timeoutMs: Long = 4_000L): Boolean {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (SystemClock.elapsedRealtime() < deadline) {
            if (cameraController.isActive) return true
            delay(50)
        }
        return cameraController.isActive
    }

    /**
     * SCHRITT 1: Foto aufnehmen und Segmentierungs-Picker anzeigen.
     * Die Kamera macht EIN scharfes Standbild. Das Foto wird angezeigt
     * und das System versucht sofort eine Automatik-Erkennung.
     */
    fun startSegCapture() {
        if (isAutoDetectRunning) { statusText = "Segmentierung läuft bereits..."; return }
        if (!hasCameraPermission) { statusText = "Kamerarecht fehlt"; return }
        uiScope.launch {
            val startedAt = SystemClock.elapsedRealtime()
            isAutoDetectRunning = true
            autoDetectProgress = 0.05f
            autoDetectPhase = "Fokussiere..."
            autoDetectStateLabel = "capturing"
            autoDetectErrorCode = "-"
            statusText = "Foto wird aufgenommen – Handy ruhig halten..."
            try {
                if (!awaitCameraReady()) {
                    statusText = "Kamera konnte nicht gestartet werden"
                    autoDetectStateLabel = "failed"; autoDetectErrorCode = "CAMERA_START_TIMEOUT"
                    return@launch
                }

                // Autofokus einschwingen lassen
                cameraController.triggerAutoFocus()
                delay(900)

                // EIN hochwertiges Standbild aufnehmen (ImageCapture – kein Video-Frame)
                autoDetectProgress = 0.25f
                autoDetectPhase = "Aufnahme (höchste Qualität)..."
                val frame = withContext(Dispatchers.Default) {
                    withTimeoutOrNull(10_000L) {
                        kotlinx.coroutines.suspendCancellableCoroutine<FrameData?> { cont ->
                            cameraController.captureStillFrame { f -> cont.resumeWith(Result.success(f)) }
                        }
                    }
                }
                if (frame == null) {
                    statusText = "Foto-Aufnahme fehlgeschlagen – Kamerarecht prüfen"
                    autoDetectStateLabel = "failed"; autoDetectErrorCode = "NO_FRAME"
                    return@launch
                }

                // Bitmap für die Anzeige im Picker
                val bmp = withContext(Dispatchers.Default) { convertFrameToBitmap(frame) }
                if (bmp == null) {
                    statusText = "Bild konnte nicht konvertiert werden"
                    autoDetectStateLabel = "failed"; autoDetectErrorCode = "BITMAP_FAIL"
                    return@launch
                }

                // Vorigen Picker-/Scan-Zustand sauber aufraeumen.
                clearSamScanResults()
                clearBoardCalibration()
                subTargetMode = false
                overlayState.subTargets.forEach { it.maskBitmap?.recycle() }
                overlayState = overlayState.copy(
                    boardTlXNorm = null,
                    boardTlYNorm = null,
                    boardTrXNorm = null,
                    boardTrYNorm = null,
                    boardBrXNorm = null,
                    boardBrYNorm = null,
                    boardBlXNorm = null,
                    boardBlYNorm = null,
                    maskBitmap = null,
                    detectedTargets = emptyList(),
                    subTargets = emptyList(),
                    samScanResults = emptyList()
                )
                nextSubTargetId = 1
                capturedSegBitmap?.takeIf { it !== bmp && !it.isRecycled }?.recycle()

                // Picker aktivieren
                capturedSegFrameRef.set(frame)
                capturedSegBitmap = bmp
                showSegmentationPicker = true
                autoDetectStateLabel = "waiting_tap"
                statusText = "Foto bereit – Vollscan läuft..."
                autoScanRequested = true

            } finally {
                autoDetectRuntimeMs = (SystemClock.elapsedRealtime() - startedAt).coerceAtLeast(0L)
                isAutoDetectRunning = false
                autoDetectProgress = 0f
                if (showSegmentationPicker && autoDetectStateLabel != "ok") autoDetectPhase = "Warte auf Tap..."
            }
        }
    }

    /**
     * SCHRITT 2: Segmentiert das Ziel an der angetippten Stelle im gespeicherten Foto.
     * Nutzt EfficientSAM mit dem Tap-Punkt als Prompt. Danach: Sub-Target-Erkennung
     * auf dem segmentierten Bereich (2. unabhängiger Schritt).
     */
    fun runSegmentationAtPoint(xNorm: Double, yNorm: Double) {
        val frame = capturedSegFrameRef.get() ?: run {
            statusText = "Kein aufgenommenes Bild – bitte erneut Auto-Erkennung starten"
            return
        }
        if (isAutoDetectRunning) return
        // Picker bleibt offen: nach erfolgreicher Stage-1-Segmentierung schalten wir
        // unten auf subTargetMode um, damit der User direkt Sub-Ziele antippen kann.
        // Beim Fehlschlag oder Cancel wird der Picker explizit geschlossen.

        uiScope.launch {
            val startedAt = SystemClock.elapsedRealtime()
            isAutoDetectRunning = true
            autoDetectProgress = 0.40f
            autoDetectPhase = "Segmentiere ${frame.width}×${frame.height} bei (${String.format(Locale.US,"%.2f",xNorm)}, ${String.format(Locale.US,"%.2f",yNorm)})..."
            autoDetectStateLabel = "running"
            autoDetectErrorCode = "-"
            statusText = "Segmentiere Zielbereich..."

            try {
                // ── Phase A: ML-Segmentierung mit Tap-Punkt als SAM-Prompt ─────
                val pointHint = xNorm to yNorm
                val segResult = withContext(Dispatchers.Default) {
                    val ml = withTimeoutOrNull(30_000L) { mlSegmenter.detectAutomatic(frame, pointHint) }
                    if (ml?.result != null) ml else detectBoardHighQuality(frame)
                }

                if (segResult.result == null) {
                    lastAutoDetect = segResult.debug
                    autoDetectStateLabel = "failed"; autoDetectErrorCode = "SEG_FAIL"
                    lastSegmentationBackend = "failed"
                    statusText = "Fehler: ${segResult.debug}. Bitte eine andere Stelle antippen."
                    // Wir lassen showSegmentationPicker = true, damit der User
                    // es nochmal an einer anderen Stelle versuchen kann.
                    return@launch
                }

                val res = segResult.result
                val dbg = segResult.debug.lowercase(Locale.ROOT)
                lastSegmentationBackend = when {
                    dbg.startsWith("sam3") -> "sam3"
                    dbg.startsWith("esam") -> "esam"
                    dbg.startsWith("sam")  -> "sam"
                    dbg.startsWith("hq")   -> "hq-heuristic"
                    else -> dbg.split(Regex("""[\s:-]"""))[0].takeIf { it.isNotBlank() } ?: "ml"
                }

                if (segResult.mask != null) {
                    // ── SAM-Maske: Zentroid berechnen + Board-Ecken löschen ──────
                    val maskW = segResult.mask.width; val maskH = segResult.mask.height
                    val maskPixels = IntArray(maskW * maskH)
                    segResult.mask.getPixels(maskPixels, 0, maskW, 0, 0, maskW, maskH)
                    var sumX = 0L; var sumY = 0L; var maskCount = 0L
                    var maskMinX = Int.MAX_VALUE; var maskMaxX = Int.MIN_VALUE
                    var maskMinY = Int.MAX_VALUE; var maskMaxY = Int.MIN_VALUE
                    maskPixels.forEachIndexed { idx, px ->
                        if (((px ushr 24) and 0xFF) > 16) {
                            val mx = idx % maskW; val my = idx / maskW
                            sumX += mx; sumY += my; maskCount++
                            if (mx < maskMinX) maskMinX = mx; if (mx > maskMaxX) maskMaxX = mx
                            if (my < maskMinY) maskMinY = my; if (my > maskMaxY) maskMaxY = my
                        }
                    }
                    if (maskCount > 0) {
                        targetXNorm = (sumX.toDouble() / maskCount / maskW).coerceIn(0.0, 1.0)
                        targetYNorm = (sumY.toDouble() / maskCount / maskH).coerceIn(0.0, 1.0)
                    }
                    // Maske IS das Target – keine Ecken nötig
                    boardTlXNorm = null; boardTlYNorm = null
                    boardTrXNorm = null; boardTrYNorm = null
                    boardBrXNorm = null; boardBrYNorm = null
                    boardBlXNorm = null; boardBlYNorm = null

                    // ── Phase B (unabhängig): Sub-Targets auf dem segmentierten Bereich ──
                    autoDetectProgress = 0.75f
                    autoDetectPhase = "Schritt 2: Erkenne Unterziele im Zielbereich..."
                    val tlXb = if (maskMinX < Int.MAX_VALUE) maskMinX.toDouble() / maskW else 0.0
                    val tlYb = if (maskMinY < Int.MAX_VALUE) maskMinY.toDouble() / maskH else 0.0
                    val brXb = if (maskMaxX > Int.MIN_VALUE) maskMaxX.toDouble() / maskW else 1.0
                    val brYb = if (maskMaxY > Int.MIN_VALUE) maskMaxY.toDouble() / maskH else 1.0
                    val targets = withContext(Dispatchers.Default) {
                        detectTargetsOnBoard(frame, tlXb, tlYb, brXb, tlYb, brXb, brYb, tlXb, brYb)
                    }

                    mainExecutor.execute {
                        overlayState = overlayState.copy(
                            boardTlXNorm = null, boardTlYNorm = null,
                            boardTrXNorm = null, boardTrYNorm = null,
                            boardBrXNorm = null, boardBrYNorm = null,
                            boardBlXNorm = null, boardBlYNorm = null,
                            maskBitmap = segResult.mask,
                            detectedTargets = targets
                        )
                    }

                    autoDetectProgress = 1.0f
                    autoDetectPhase = "Abgeschlossen"
                    autoDetectStateLabel = "ok"
                    updateOnboardingStep()
                    val confText = format3(res.confidence)
                    lastAutoDetect = "${segResult.debug} | mask ${maskCount}px | ${targets.size} sub-targets"
                    // Stage 1 fertig → Multi-Tap-Modus aktivieren, Picker offen lassen.
                    // User kann jetzt einzelne Sub-Ziele direkt auf dem Foto antippen.
                    subTargetMode = true
                    nextSubTargetId = 1
                    statusText = if (targets.isNotEmpty()) {
                        targetXNorm = targets[0].first; targetYNorm = targets[0].second
                        "✓ Platte segmentiert ($lastSegmentationBackend) – ${targets.size} Auto-Hint(s) gefunden. Jetzt Sub-Ziele antippen oder 'Fertig'."
                    } else {
                        "✓ Platte segmentiert ($lastSegmentationBackend conf $confText) – Sub-Ziele antippen oder 'Fertig' druecken."
                    }

                } else {
                    // ── Heuristik-Fallback: Bounding-Box + Ecken setzen ──────────
                    setBoardCorner(0, res.tlX, res.tlY)
                    setBoardCorner(1, res.trX, res.trY)
                    setBoardCorner(2, res.brX, res.brY)
                    setBoardCorner(3, res.blX, res.blY)

                    if (!isBoardGeometryValid()) {
                        lastAutoDetect = "Ungültige Geometrie: ${segResult.debug}"
                        autoDetectStateLabel = "failed"; autoDetectErrorCode = "GEOMETRY_INVALID"
                        statusText = "Erkannte Form kein gültiges Viereck – erneut antippen"
                        clearBoardCalibration()
                        // Picker bleibt offen
                        return@launch
                    }

                    mainExecutor.execute {
                        overlayState = overlayState.copy(
                            boardTlXNorm = res.tlX, boardTlYNorm = res.tlY,
                            boardTrXNorm = res.trX, boardTrYNorm = res.trY,
                            boardBrXNorm = res.brX, boardBrYNorm = res.brY,
                            boardBlXNorm = res.blX, boardBlYNorm = res.blY,
                            maskBitmap = null
                        )
                    }

                    // ── Phase B: Sub-Targets auf der erkannten Platte ──────────
                    autoDetectProgress = 0.80f
                    autoDetectPhase = "Schritt 2: Erkenne Targets auf der Platte..."
                    val targets = withContext(Dispatchers.Default) {
                        detectTargetsOnBoard(frame, res.tlX, res.tlY, res.trX, res.trY, res.brX, res.brY, res.blX, res.blY)
                    }

                    autoDetectProgress = 1.0f
                    autoDetectPhase = "Abgeschlossen"
                    autoDetectStateLabel = "ok"
                    updateOnboardingStep()
                    val confText = format3(res.confidence)
                    lastAutoDetect = "${segResult.debug} | ${targets.size} targets"
                    // Heuristik-Fallback liefert keine Pixel-Maske → Multi-Tap (Stage 2)
                    // braucht aber genau diese. Picker hier schliessen.
                    showSegmentationPicker = false
                    statusText = if (targets.isEmpty()) {
                        "Heuristik-Box (FloodFill lieferte keine Maske)"
                    } else {
                        targetXNorm = targets[0].first; targetYNorm = targets[0].second
                        "Heuristik-Box | ${targets.size} Unterziel(e)"
                    }
                }
            } finally {
                autoDetectRuntimeMs = (SystemClock.elapsedRealtime() - startedAt).coerceAtLeast(0L)
                delay(200)
                isAutoDetectRunning = false
                autoDetectProgress = 0f
                autoDetectPhase = if (autoDetectStateLabel == "ok") "Abgeschlossen (${autoDetectRuntimeMs}ms)" else "Beendet (${autoDetectRuntimeMs}ms)"
            }
        }
    }

    fun refineAndAcceptScanMask(rawMask: Bitmap, xNorm: Double, yNorm: Double) {
        val photoBmp = capturedSegBitmap ?: run {
            statusText = "Kein Foto vorhanden"
            return
        }
        val frame = capturedSegFrameRef.get()
        if (isAutoDetectRunning || isSubTargetSegmenting) return

        uiScope.launch {
            val startedAt = SystemClock.elapsedRealtime()
            isAutoDetectRunning = true
            autoDetectProgress = 0.85f
            autoDetectPhase = "Verfeinere angetippten Bereich..."
            autoDetectStateLabel = "running"
            try {
                val sx = (xNorm * photoBmp.width).toInt().coerceIn(0, photoBmp.width - 1)
                val sy = (yNorm * photoBmp.height).toInt().coerceIn(0, photoBmp.height - 1)
                val refinedMask = withContext(Dispatchers.Default) {
                    SamOnnxHelper.runEfficientSamPointInMask(photoBmp, rawMask, xNorm, yNorm)
                        ?: runCatching {
                            LocalSegmenter.segmentFromPointWithinMask(
                                bitmap = photoBmp,
                                stage1Mask = rawMask,
                                seedX = sx,
                                seedY = sy,
                                tolerance = 28,
                                maxDim = 1024
                            ).maskBitmap
                        }.getOrNull()
                }

                val finalMask = refinedMask ?: rawMask
                val maskW = finalMask.width
                val maskH = finalMask.height
                val pixels = IntArray(maskW * maskH)
                finalMask.getPixels(pixels, 0, maskW, 0, 0, maskW, maskH)
                var sumX = 0L
                var sumY = 0L
                var count = 0L
                var minX = Int.MAX_VALUE
                var minY = Int.MAX_VALUE
                var maxX = Int.MIN_VALUE
                var maxY = Int.MIN_VALUE
                pixels.forEachIndexed { idx, px ->
                    if (((px ushr 24) and 0xFF) > 16) {
                        val mx = idx % maskW
                        val my = idx / maskW
                        sumX += mx
                        sumY += my
                        count++
                        if (mx < minX) minX = mx
                        if (my < minY) minY = my
                        if (mx > maxX) maxX = mx
                        if (my > maxY) maxY = my
                    }
                }

                val areaNorm = count.toDouble() / (maskW.toLong() * maskH.toLong()).coerceAtLeast(1L).toDouble()
                if (count <= 0L || areaNorm < 0.0008) {
                    if (refinedMask != null && refinedMask !== rawMask && !refinedMask.isRecycled) refinedMask.recycle()
                    statusText = "Der angetippte Teil war zu klein oder leer - bitte genauer auf den Bunker tippen"
                    return@launch
                }

                val tlXb = if (minX < Int.MAX_VALUE) minX.toDouble() / maskW else 0.0
                val tlYb = if (minY < Int.MAX_VALUE) minY.toDouble() / maskH else 0.0
                val brXb = if (maxX > Int.MIN_VALUE) maxX.toDouble() / maskW else 1.0
                val brYb = if (maxY > Int.MIN_VALUE) maxY.toDouble() / maskH else 1.0
                val targets = if (frame != null) {
                    withContext(Dispatchers.Default) {
                        detectTargetsOnBoard(frame, tlXb, tlYb, brXb, tlYb, brXb, brYb, tlXb, brYb)
                    }
                } else {
                    emptyList()
                }

                if (finalMask === rawMask) clearSamScanResults(keep = rawMask) else clearSamScanResults()
                overlayState = overlayState.copy(
                    boardTlXNorm = null,
                    boardTlYNorm = null,
                    boardTrXNorm = null,
                    boardTrYNorm = null,
                    boardBrXNorm = null,
                    boardBrYNorm = null,
                    boardBlXNorm = null,
                    boardBlYNorm = null,
                    maskBitmap = finalMask,
                    detectedTargets = targets,
                    samScanResults = emptyList()
                )
                targetXNorm = (sumX.toDouble() / count / maskW).coerceIn(0.0, 1.0)
                targetYNorm = (sumY.toDouble() / count / maskH).coerceIn(0.0, 1.0)
                overlayState = overlayState.copy(targetXNorm = targetXNorm, targetYNorm = targetYNorm)
                lastSegmentationBackend = if (refinedMask != null) "scan-refined" else "scan-picked"
                lastAutoDetect = "scan-pick area=${format3(areaNorm)} hints=${targets.size}"
                subTargetMode = true
                nextSubTargetId = 1
                autoDetectStateLabel = "ok"
                autoDetectProgress = 1.0f
                autoDetectPhase = "Auswahl übernommen"
                updateOnboardingStep()
                statusText = if (targets.isNotEmpty()) {
                    "✓ Nur der angetippte Bereich wurde übernommen. Jetzt Sub-Ziele setzen oder 'Fertig'."
                } else {
                    "✓ Auswahl auf den angetippten Bereich verfeinert. Jetzt Sub-Ziele setzen oder 'Fertig'."
                }
            } catch (t: Throwable) {
                android.util.Log.e("ScanPick", "refine failed", t)
                statusText = "Auswahl konnte nicht verfeinert werden: ${t.message ?: t::class.java.simpleName}"
                autoDetectStateLabel = "failed"
            } finally {
                autoDetectRuntimeMs = (SystemClock.elapsedRealtime() - startedAt).coerceAtLeast(0L)
                isAutoDetectRunning = false
                autoDetectProgress = 0f
                if (autoDetectStateLabel != "ok") autoDetectPhase = ""
            }
        }
    }

    /** Kompakte Diagnose fuer den statusText (Segmentier-Backend). */
    fun samDiagShort(): String = "Backend: OpenCV-FloodFill"

    /**
     * STAGE 2 Multi-Tap: nimmt einen Tap auf das aufgenommene Foto, ruft
     * EfficientSAM restriktiv auf die Stage-1-Maske auf und legt das Ergebnis
     * als neues TargetEntry in overlayState.subTargets ab. Das Foto bleibt
     * sichtbar damit der User direkt das naechste Ziel antippen kann.
     */
    fun addSubTargetAtPoint(xNorm: Double, yNorm: Double) {
        val photoBmp = capturedSegBitmap ?: run {
            statusText = "Kein Foto vorhanden – bitte zuerst Plattenerkennung starten"
            return
        }
        val stage1Mask = overlayState.maskBitmap ?: run {
            statusText = "Zuerst die Platte segmentieren, dann Multi-Tap fuer Ziele"
            return
        }
        if (isSubTargetSegmenting) return
        isSubTargetSegmenting = true
        val id = nextSubTargetId
        uiScope.launch {
            try {
                statusText = "Segmentiere Ziel #$id..."
                val seg = withContext(Dispatchers.Default) {
                    val sx = (xNorm * photoBmp.width).toInt().coerceIn(0, photoBmp.width - 1)
                    val sy = (yNorm * photoBmp.height).toInt().coerceIn(0, photoBmp.height - 1)
                    LocalSegmenter.segmentFromPointWithinMask(photoBmp, stage1Mask, sx, sy)
                }
                val mask = seg.maskBitmap
                val mw = mask.width; val mh = mask.height
                val areaNorm = seg.pixelArea.toDouble() / (mw * mh).coerceAtLeast(1)
                if (seg.pixelArea <= 0 || areaNorm < 0.0005) {
                    mask.recycle()
                    statusText = "Ziel #$id zu klein - deutlicher antippen"
                    return@launch
                }
                val entry = TargetEntry(
                    id = id,
                    centerXNorm = xNorm,
                    centerYNorm = yNorm,
                    maskBitmap = mask,
                    areaNorm = areaNorm
                )
                nextSubTargetId = id + 1
                overlayState = overlayState.copy(subTargets = overlayState.subTargets + entry)
                updateFocusFromSelectedTargets()
                val pct = String.format(Locale.US, "%.2f", areaNorm * 100)
                statusText = "OK Ziel #$id erfasst (${pct}% Flaeche). Naechstes antippen oder Fertig druecken."
            } catch (oom: OutOfMemoryError) {
                System.gc()
                statusText = "Stage 2: Speicher knapp - Foto zu gross. Bitte 'X' druecken und neu aufnehmen."
                android.util.Log.e("AddSubTarget", "OOM at Stage 2 segmentation", oom)
            } catch (t: Throwable) {
                statusText = "Stage 2 Fehler: ${t.message ?: t::class.java.simpleName}"
                android.util.Log.e("AddSubTarget", "Stage 2 failed", t)
            } finally {
                isSubTargetSegmenting = false
            }
        }
    }

    fun removeLastSubTarget() {
        val list = overlayState.subTargets
        if (list.isEmpty()) return
        list.last().maskBitmap?.recycle()
        val newList = list.dropLast(1)
        overlayState = overlayState.copy(subTargets = newList)
        if (newList.isEmpty()) nextSubTargetId = 1
        updateFocusFromSelectedTargets()
        statusText = if (newList.isEmpty()) "Alle Sub-Ziele entfernt" else "Letztes Ziel entfernt (${newList.size} bleiben)"
    }

    fun clearSubTargets() {
        if (overlayState.subTargets.isEmpty()) return
        overlayState.subTargets.forEach { it.maskBitmap?.recycle() }
        overlayState = overlayState.copy(subTargets = emptyList())
        nextSubTargetId = 1
        updateFocusFromSelectedTargets()
        statusText = "Sub-Ziele zurueckgesetzt"
    }

    /**
     * STAGE 2 - NEUES FOTO: Nutzer hat die Sub-Ziele jetzt erst angebracht
     * (Sticker / Pepper-Popper / etc.). Wir machen ein zweites Standbild und
     * verwenden weiterhin die Stage-1-Maske. Voraussetzung: Handy wurde
     * zwischen Stage 1 und Stage 2 NICHT bewegt - sonst passt die Maske nicht.
     * Bereits angelegte Sub-Ziele werden verworfen, weil sie aufs alte Foto
     * referenziert sind.
     */
    fun recaptureForStage2() {
        if (isAutoDetectRunning || isSubTargetSegmenting) {
            statusText = "Vorgang laeuft noch - kurz warten"
            return
        }
        if (!hasCameraPermission) { statusText = "Kamerarecht fehlt"; return }
        if (overlayState.maskBitmap == null) {
            statusText = "Stage 1 noch nicht segmentiert"
            return
        }
        uiScope.launch {
            val startedAt = SystemClock.elapsedRealtime()
            isAutoDetectRunning = true
            autoDetectProgress = 0.05f
            autoDetectPhase = "Stage 2: Neues Foto..."
            autoDetectStateLabel = "capturing"
            statusText = "Stage-2-Foto: Handy ruhig halten..."
            try {
                if (!awaitCameraReady()) {
                    statusText = "Stage-2-Kamera konnte nicht gestartet werden"
                    autoDetectStateLabel = "failed"; autoDetectErrorCode = "CAMERA_START_TIMEOUT_S2"
                    return@launch
                }
                cameraController.triggerAutoFocus()
                delay(900)
                autoDetectProgress = 0.30f
                autoDetectPhase = "Stage 2: Aufnahme..."
                val frame = withContext(Dispatchers.Default) {
                    withTimeoutOrNull(10_000L) {
                        kotlinx.coroutines.suspendCancellableCoroutine<FrameData?> { cont ->
                            cameraController.captureStillFrame { f -> cont.resumeWith(Result.success(f)) }
                        }
                    }
                }
                if (frame == null) {
                    statusText = "Stage-2-Foto fehlgeschlagen - bitte erneut"
                    autoDetectStateLabel = "failed"; autoDetectErrorCode = "NO_FRAME_S2"
                    return@launch
                }
                val bmp = withContext(Dispatchers.Default) { convertFrameToBitmap(frame) }
                if (bmp == null) {
                    statusText = "Stage-2-Bild konnte nicht konvertiert werden"
                    autoDetectStateLabel = "failed"; autoDetectErrorCode = "BITMAP_FAIL_S2"
                    return@launch
                }

                // Sub-Targets verwerfen (basierten aufs alte Foto)
                overlayState.subTargets.forEach { it.maskBitmap?.recycle() }
                overlayState = overlayState.copy(subTargets = emptyList())
                nextSubTargetId = 1

                // Altes Foto recyclen, neues setzen. Stage-1-Maske bleibt erhalten.
                capturedSegBitmap?.recycle()
                capturedSegBitmap = bmp
                capturedSegFrameRef.set(frame)

                autoDetectStateLabel = "waiting_tap"
                statusText = "Stage-2-Foto bereit. Sub-Ziele jetzt antippen."
            } catch (t: Throwable) {
                android.util.Log.e("Stage2Recapture", "failed", t)
                statusText = "Stage-2-Foto-Fehler: ${t.message ?: t::class.java.simpleName}"
                autoDetectStateLabel = "failed"; autoDetectErrorCode = "S2_EXC"
            } finally {
                autoDetectRuntimeMs = (SystemClock.elapsedRealtime() - startedAt).coerceAtLeast(0L)
                isAutoDetectRunning = false
                autoDetectProgress = 0f
            }
        }
    }
    /** Rückwärtskompatibel: Startet Schritt 1 (Foto aufnehmen). */
    fun runAutoBoardDetection(tapXNorm: Double? = null, tapYNorm: Double? = null) {
        startSegCapture()
    }

    /**
     * TEST-MODUS: Segmentiert das ganze Bild mit einem 3x3 Grid und zeigt alle Segmente.
     * Der Nutzer kann dann eines durch Antippen auswählen.
     */
    fun runSamScan() {
        val bmp = capturedSegBitmap ?: run {
            statusText = "Kein Foto vorhanden"
            return
        }
        if (isSamScanRunning) return
        uiScope.launch {
            try {
                isSamScanRunning = true
                statusText = "Vollscan läuft..."
                clearSamScanResults()
                overlayState = overlayState.copy(maskBitmap = null, samScanResults = emptyList())
                val results = mutableListOf<Bitmap>()
                val resultRegions = mutableListOf<AutoBoardResult>()
                val fullImageSegments = withContext(Dispatchers.Default) {
                    LocalSegmenter.segmentWholeImage(
                        bitmap = bmp,
                        maxDim = 480,
                        minAreaNorm = 0.0012,
                        maxAreaNorm = 0.88,
                        maxRegions = 28
                    )
                }

                for ((index, segment) in fullImageSegments.withIndex()) {
                    autoDetectPhase = "Vollbild-Segmentierung ${index + 1}/${fullImageSegments.size}"
                    val mask = segment.maskBitmap
                    val region = mlSegmenter.maskToRegion(mask)
                    if (region == null) {
                        if (!mask.isRecycled) mask.recycle()
                        continue
                    }
                    val board = mlSegmenter.segmentRegionToAutoBoardResult(region, mask.width, mask.height)
                    val duplicate = resultRegions.any { rectIou(it, board) >= 0.58 }
                    if (duplicate) {
                        if (!mask.isRecycled) mask.recycle()
                        continue
                    }
                    resultRegions += board
                    results += mask
                }

                val efficientSamReady = withContext(Dispatchers.Default) {
                    if (!SamOnnxHelper.isEfficientSamAvailable()) {
                        runCatching { SamOnnxHelper.initEfficientSam(context) }
                    }
                    SamOnnxHelper.isEfficientSamAvailable()
                }
                if (results.size < 8) {
                    val scanPoints = buildFullImageScanPoints(bmp.width, bmp.height)
                    var count = 0
                    val total = scanPoints.size
                    for ((gx, gy) in scanPoints) {
                        count++
                        autoDetectPhase = "Prompt-Scan $count/$total"
                        val mask = withContext(Dispatchers.Default) {
                            if (efficientSamReady) {
                                SamOnnxHelper.runEfficientSamPoint(bmp, gx, gy)
                            } else {
                                val sx = (gx * bmp.width).toInt().coerceIn(0, bmp.width - 1)
                                val sy = (gy * bmp.height).toInt().coerceIn(0, bmp.height - 1)
                                runCatching { LocalSegmenter.segmentFromPoint(bmp, sx, sy, tolerance = 56, maxDim = 768).maskBitmap }.getOrNull()
                            }
                        }
                        if (mask == null) continue

                        val region = mlSegmenter.maskToRegion(mask)
                        if (region == null) {
                            if (!mask.isRecycled) mask.recycle()
                            continue
                        }

                        val areaNorm = region.area.toDouble() / (mask.width.toLong() * mask.height.toLong()).coerceAtLeast(1L).toDouble()
                        if (areaNorm !in 0.0015..0.95) {
                            if (!mask.isRecycled) mask.recycle()
                            continue
                        }

                        val board = mlSegmenter.segmentRegionToAutoBoardResult(region, mask.width, mask.height)
                        val duplicate = resultRegions.any { rectIou(it, board) >= 0.60 }
                        if (duplicate) {
                            if (!mask.isRecycled) mask.recycle()
                            continue
                        }

                        resultRegions += board
                        results += mask
                        if (results.size >= 20) break
                    }
                }
                samScanResults = results
                overlayState = overlayState.copy(samScanResults = results)
                autoDetectPhase = ""
                lastSegmentationBackend = if (efficientSamReady) "sam-scan" else "floodfill-scan"
                statusText = if (results.isEmpty()) {
                    "Vollscan fand nichts - bitte Platte direkt antippen"
                } else {
                    "${results.size} Segment(e) gefunden - Segment antippen"
                }
            } catch (t: Throwable) {
                android.util.Log.e("SamScan", "scan failed", t)
                statusText = "Vollscan fehlgeschlagen: ${t.message ?: t::class.java.simpleName}"
            } finally {
                isSamScanRunning = false
            }
        }
    }

    LaunchedEffect(showSegmentationPicker, autoScanRequested, isSamScanRunning, capturedSegBitmap) {
        if (showSegmentationPicker && autoScanRequested && !isSamScanRunning && capturedSegBitmap != null) {
            autoScanRequested = false
            runSamScan()
        }
    }

    fun syncRuntimePermissions() {
        hasMicPermission = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        hasCameraPermission = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { result ->
        hasMicPermission = result[Manifest.permission.RECORD_AUDIO] == true
        hasCameraPermission = result[Manifest.permission.CAMERA] == true
        syncRuntimePermissions()
        if (hasMicPermission && hasCameraPermission) {
            updateOnboardingStep()
            if (pendingStartAfterPermission && !isRunning) {
                pendingStartAfterPermission = false; statusText = "Rechte erteilt, starte..."; startPipeline()
            }
        } else {
            pendingStartAfterPermission = false
            statusText = "Rechte fuer Mikrofon und Kamera erforderlich"
            updateOnboardingStep()
        }
    }

    LaunchedEffect(previewView, hasCameraPermission, showSegmentationPicker, isAutoDetectRunning, cameraExpanded, isRunning, runtimeMemoryProfile.lowMemoryMode) {
        val view = previewView
        val shouldBind = hasCameraPermission && (showSegmentationPicker || isAutoDetectRunning || cameraExpanded || isRunning)
        if (!hasCameraPermission) { statusText = "Kamera-Recht fehlt"; cameraController.stop(); return@LaunchedEffect }
        if (!shouldBind) {
            cameraController.stop()
            previewView = null
            firstFrameSeen = false
            return@LaunchedEffect
        }
        if (view == null) { statusText = "Warte auf Kamera-Preview..."; cameraController.stop(); return@LaunchedEffect }
        cameraController.start(
            lifecycleOwner = lifecycleOwner,
            previewView = view,
            lowMemoryMode = runtimeMemoryProfile.lowMemoryMode,
            onFrame = { frame ->
                ringBuffer.add(frame)
                latestFrameForAutoDetect.set(frame)
                val m = motionAnalyzer.analyze(frame)
                latestMotionScore.set(m.score)
                mainExecutor.execute {
                    motionState = m
                    overlayState = overlayState.copy(
                        targetXNorm = targetXNorm, targetYNorm = targetYNorm,
                        boardTlXNorm = boardTlXNorm, boardTlYNorm = boardTlYNorm,
                        boardTrXNorm = boardTrXNorm, boardTrYNorm = boardTrYNorm,
                        boardBrXNorm = boardBrXNorm, boardBrYNorm = boardBrYNorm,
                        boardBlXNorm = boardBlXNorm, boardBlYNorm = boardBlYNorm
                    )
                    ringSize = ringBuffer.size(); frameTick += 1
                    lastDetectorSource = m.source; lastYolo = format3(m.yoloConfidence)
                    if (!firstFrameSeen) {
                        firstFrameSeen = true
                        statusText = if (isRunning && firstAudioSeen) "Pipeline aktiv (Audio+Kamera)" else "Kamera bereit - Start fuer Audio/Trigger"
                    }
                }
            },
            onError = { msg -> mainExecutor.execute { statusText = msg } }
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            saveProfiles(); detector.stop(); cameraController.release(); exporter.shutdown(); mlSegmenter.close()
            tts.stop(); tts.shutdown()
        }
    }

    // rememberUpdatedState, damit der LifecycleObserver auf den AKTUELLEN Wert
    // zugreift und nicht den beim ersten Compose gecachten - sonst weiss er
    // beim Stoppen nicht ob die Pipeline gerade laeuft.
    val isRunningLatest = rememberUpdatedState(isRunning)
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> {
                    syncRuntimePermissions(); updateOnboardingStep()
                }
                Lifecycle.Event.ON_STOP -> {
                    // Display aus / App im Hintergrund: Audio-Detector laeuft sonst
                    // auf seinem Background-Thread weiter und frisst Akku. CameraX
                    // pausiert via Lifecycle automatisch, nur Audio muss explizit
                    // gestoppt werden. UI-State synchronisieren, damit beim
                    // Wiederkommen klar ist dass die Pipeline pausiert wurde.
                    if (isRunningLatest.value) {
                        try { detector.stop() } catch (_: Throwable) {}
                        cameraController.analysisEnabled = false
                        isRunning = false
                        statusText = "Pipeline pausiert (Hintergrund) - Start zum Fortsetzen"
                    }
                }
                else -> { /* no-op */ }
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    LaunchedEffect(hitMarkerToken) {
        if (hitMarkerToken == 0L) return@LaunchedEffect
        delay(1_200)
        overlayState = overlayState.copy(hitXNorm = null, hitYNorm = null, hitWasValid = null)
    }

    val profileReady = playerName.isNotBlank() && weaponProfiles.isNotEmpty()
    val permissionReady = hasMicPermission && hasCameraPermission
    val boardReady = isBoardComplete() && isBoardGeometryValid()
    val calibrationReady = calibrationShots >= 5
    val onboardingReady = profileReady && permissionReady && boardReady && calibrationReady
    val setupLayerActive = !onboardingDone && !setupLayerDismissed
    val segmentationFullscreen = showSegmentationPicker || isAutoDetectRunning
    val cameraShouldShow = segmentationFullscreen || cameraExpanded
    val cameraShouldBeBound = hasCameraPermission && (cameraShouldShow || isRunning)
    val cameraReady = hasCameraPermission && firstFrameSeen
    val audioReady = hasMicPermission && firstAudioSeen && isRunning
    val loadoutReady = player1WeaponId.isNotBlank() && (playerCount == 1 || player2WeaponId.isNotBlank())

    LaunchedEffect(isRunning, cameraShouldBeBound) {
        cameraController.analysisEnabled = isRunning && cameraShouldBeBound
    }

    Box(modifier = modifier.fillMaxSize().background(tactDark)) {
        Column(modifier = Modifier.fillMaxSize()) {
            // ── Kompakter Header ───────────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth().background(tactGray).padding(horizontal = 12.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Image(painter = painterResource(id = R.drawable.gross_logo), contentDescription = stringResource(id = R.string.app_logo_desc), modifier = Modifier.height(32.dp), contentScale = ContentScale.Fit)
                Spacer(modifier = Modifier.weight(1f))
                
                // Kamera-Toggle
                val cameraForced = showSegmentationPicker || isAutoDetectRunning
                androidx.compose.material3.IconButton(
                    onClick = { cameraExpanded = !cameraExpanded },
                    enabled = !cameraForced,
                    modifier = Modifier.size(32.dp).background(if (cameraExpanded || cameraForced) tactAccent else tactGrayLight, RoundedCornerShape(4.dp))
                ) {
                    Text(if (cameraExpanded || cameraForced) "\uD83D\uDCF7" else "\uD83D\uDCF7", fontSize = 16.sp)
                }

                Spacer(modifier = Modifier.width(4.dp))
                val badgeColor = when { isRunning && cameraReady && audioReady -> tactAccent; isRunning || cameraReady -> tactAmber; else -> tactTextDim }
                Box(modifier = Modifier.background(badgeColor.copy(alpha = 0.15f), RoundedCornerShape(2.dp)).border(1.dp, badgeColor.copy(alpha = 0.3f), RoundedCornerShape(2.dp)).padding(horizontal = 8.dp, vertical = 4.dp)) {
                    Text(text = when { isRunning && cameraReady && audioReady -> "ONLINE"; isRunning -> "AUDIO ONLY"; cameraReady -> "OPTIC ONLY"; else -> "STANDBY" },
                        style = MaterialTheme.typography.labelSmall, color = badgeColor, fontWeight = FontWeight.ExtraBold, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
                }
            }

            // ── Kamera ───────────────────────────────────────────────────────
            if (cameraShouldShow) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .then(if (segmentationFullscreen) Modifier.weight(1f) else Modifier.weight(0.40f))
                        .background(Color.Black)
                        .border(1.dp, tactBorder)
                        // ── Tap: Foto aufnehmen → Picker anzeigen (Schritt 1) ──
                        .pointerInput(isAutoDetectRunning, showSegmentationPicker) {
                            detectTapGestures(
                                onTap = { _ ->
                                    if (!isAutoDetectRunning && !showSegmentationPicker && hasCameraPermission) {
                                        startSegCapture()
                                    }
                                }
                            )
                        }
                        // ── Drag: Board-Ecken verschieben ────────────────────────
                        .pointerInput(boardTlXNorm, boardTlYNorm, boardTrXNorm, boardTrYNorm, boardBrXNorm, boardBrYNorm, boardBlXNorm, boardBlYNorm) {
                            detectDragGestures(
                                onDragStart = { start ->
                                    if (size.width <= 0 || size.height <= 0) return@detectDragGestures
                                    val xNorm = (start.x / size.width).toDouble().coerceIn(0.0, 1.0)
                                    val yNorm = (start.y / size.height).toDouble().coerceIn(0.0, 1.0)
                                    activeDraggedCorner = nearestBoardCornerIndex(xNorm, yNorm)
                                },
                                onDrag = { change, _ ->
                                    val cornerIndex = activeDraggedCorner ?: return@detectDragGestures
                                    if (size.width <= 0 || size.height <= 0) return@detectDragGestures
                                    val xNorm = (change.position.x / size.width).toDouble().coerceIn(0.0, 1.0)
                                    val yNorm = (change.position.y / size.height).toDouble().coerceIn(0.0, 1.0)
                                    setBoardCorner(cornerIndex, xNorm, yNorm); updateOnboardingStep()
                                    statusText = "Ecke ${cornerIndex + 1} wird verschoben"
                                    change.consume()
                                },
                                onDragEnd = { if (activeDraggedCorner != null) statusText = "Ecke angepasst"; activeDraggedCorner = null },
                                onDragCancel = { activeDraggedCorner = null }
                            )
                        }
                ) {
                AndroidView(
                    modifier = Modifier.fillMaxSize(),
                    factory = { ctx ->
                        PreviewView(ctx).apply {
                            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                            scaleType = PreviewView.ScaleType.FILL_CENTER
                        }.also { previewView = it }
                    },
                    update = { previewView = it }
                )
                DetectionOverlay(state = overlayState, modifier = Modifier.fillMaxSize())

                // ── Schritt-2-Picker: Foto anzeigen, Nutzer tippt Ziel an ───
                if (showSegmentationPicker) {
                    val pickerBmp = capturedSegBitmap
                    if (pickerBmp != null) {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .pointerInput(showSegmentationPicker, subTargetMode, samScanResults) {
                                    detectTapGestures(
                                        onTap = { offset ->
                                            if (isAutoDetectRunning || isSubTargetSegmenting || isSamScanRunning) return@detectTapGestures
                                            val xN = if (size.width > 0) (offset.x / size.width).toDouble().coerceIn(0.0, 1.0) else 0.5
                                            val yN = if (size.height > 0) (offset.y / size.height).toDouble().coerceIn(0.0, 1.0) else 0.5

                                            if (samScanResults.isNotEmpty()) {
                                                // ML-Scan Auswahl-Modus: pruefen welches Segment getroffen wurde
                                                var hitIdx = -1
                                                for (i in samScanResults.indices.reversed()) {
                                                    val m = samScanResults[i]
                                                    val px = (xN * m.width).toInt().coerceIn(0, m.width - 1)
                                                    val py = (yN * m.height).toInt().coerceIn(0, m.height - 1)
                                                    if (((m.getPixel(px, py) ushr 24) and 0xFF) > 16) { hitIdx = i; break }
                                                }
                                                if (hitIdx >= 0) {
                                                    val chosen = samScanResults[hitIdx]
                                                    refineAndAcceptScanMask(chosen, xN, yN)
                                                    return@detectTapGestures
                                                }
                                                // Wenn nichts getroffen wurde, Scan beenden und normal weiter
                                                clearSamScanResults()
                                                statusText = "Kein Segment getroffen. Bitte direkt auf den gewünschten Bereich tippen."
                                                return@detectTapGestures
                                            }

                                            if (subTargetMode) {
                                                addSubTargetAtPoint(xN, yN)
                                            } else {
                                                runSegmentationAtPoint(xN, yN)
                                            }
                                        }
                                    )
                                }
                        ) {
                            Image(
                                bitmap = pickerBmp.asImageBitmap(),
                                contentDescription = "Aufgenommenes Foto – Ziel antippen",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.FillBounds
                            )
                            DetectionOverlay(
                                state = OverlayViewState(
                                    maskBitmap = overlayState.maskBitmap,
                                    detectedTargets = overlayState.detectedTargets,
                                    subTargets = overlayState.subTargets,
                                    samScanResults = samScanResults
                                ),
                                modifier = Modifier.fillMaxSize()
                            )
                            androidx.compose.material3.Surface(
                                modifier = Modifier.align(Alignment.BottomStart).padding(10.dp),
                                color = Color(0xDD000000),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Column(
                                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                                    horizontalAlignment = Alignment.Start,
                                    verticalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    when {
                                        samScanResults.isNotEmpty() -> {
                                            Text(
                                                "Vollscan: ${samScanResults.size} Treffer",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = Color.White,
                                                fontWeight = FontWeight.Bold
                                            )
                                            Text(
                                                "Segment antippen = nur der angetippte Teil wird übernommen.",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = Color(0xFFBBBBBB)
                                            )
                                        }
                                        subTargetMode -> {
                                            val count = overlayState.subTargets.size
                                            Text(
                                                if (isSubTargetSegmenting) "Segmentiere Ziel #${nextSubTargetId}..."
                                                else "Stage 2: einzelne Ziele antippen",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = Color.White,
                                                fontWeight = FontWeight.Bold
                                            )
                                            Text(
                                                "Erfasst: $count Ziel(e). Weiter antippen oder oben links 'Fertig'.",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = Color(0xFFBBBBBB)
                                            )
                                            if (count > 0) {
                                                Row(
                                                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                                                    verticalAlignment = Alignment.CenterVertically
                                                ) {
                                                    Button(
                                                        onClick = { removeLastSubTarget() },
                                                        colors = ButtonDefaults.buttonColors(containerColor = Color(0x99554000)),
                                                        contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp, vertical = 4.dp)
                                                    ) { Text("Letztes zurueck", fontSize = 12.sp, color = Color.White) }
                                                    Button(
                                                        onClick = { clearSubTargets() },
                                                        colors = ButtonDefaults.buttonColors(containerColor = Color(0x99550000)),
                                                        contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp, vertical = 4.dp)
                                                    ) { Text("Alle loeschen", fontSize = 12.sp, color = Color.White) }
                                                }
                                            }
                                        }
                                        else -> {
                                            Text(
                                                "Stage 1: Platte antippen",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = Color.White,
                                                fontWeight = FontWeight.Bold
                                            )
                                            Text(
                                                "Oder oben 'Vollscan' nutzen",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = Color(0xFFBBBBBB)
                                            )
                                        }
                                    }
                                }
                            }
                            // Fertig-Button (nur im Multi-Tap-Modus, oben links)
                            if (subTargetMode) {
                                Button(
                                    modifier = Modifier.align(Alignment.TopStart).padding(8.dp),
                                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xCC2E7D32)),
                                    onClick = {
                                        subTargetMode = false
                                        updateFocusFromSelectedTargets()
                                        showSegmentationPicker = false
                                        cameraExpanded = false
                                        capturedSegBitmap?.recycle()
                                        capturedSegBitmap = null
                                        capturedSegFrameRef.set(null)
                                        autoDetectStateLabel = "idle"
                                        autoDetectPhase = ""
                                        val n = overlayState.subTargets.size
                                        statusText = if (n == 0) "Stage 2 beendet - keine Sub-Ziele angelegt"
                                                     else "OK: $n Sub-Ziel(e) gespeichert - bereit fuer Drills"
                                    }
                                ) { Text("OK Fertig", color = Color.White) }
                            }
                            // Neues-Foto-fuer-Stage-2-Button (oben Mitte) - User hat die
                            // Sub-Ziele erst NACH der Plattenerkennung angebracht und braucht
                            // ein zweites Foto. Maske bleibt, Sub-Targets werden verworfen.
                            if (subTargetMode) {
                                Button(
                                    modifier = Modifier.align(Alignment.TopCenter).padding(8.dp),
                                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xCC1565C0)),
                                    enabled = !isAutoDetectRunning && !isSubTargetSegmenting,
                                    contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                                    onClick = { recaptureForStage2() }
                                ) {
                                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                        Text(
                                            if (isAutoDetectRunning) "Foto..." else "Neues Foto fuer Stage 2",
                                            color = Color.White,
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold,
                                            maxLines = 1
                                        )
                                        Text(
                                            "Handy NICHT bewegen",
                                            color = Color(0xFFFFE082),
                                            fontSize = 9.sp,
                                            maxLines = 1
                                        )
                                    }
                                }
                            } else {
                                // Stage 1 Buttons
                                Row(modifier = Modifier.align(Alignment.TopCenter).padding(8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(
                                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xCC5E35B1)),
                                        enabled = !isAutoDetectRunning && !isSamScanRunning,
                                        onClick = { runSamScan() }
                                    ) {
                                        Text(if (isSamScanRunning) "Vollscan..." else "Vollscan", color = Color.White, fontSize = 12.sp)
                                    }
                                }
                            }
                            if (subTargetMode || isSamScanRunning || isSubTargetSegmenting || samScanResults.isNotEmpty()) {
                                androidx.compose.material3.Surface(
                                    modifier = Modifier.align(Alignment.BottomStart).padding(8.dp),
                                    color = Color(0x99000000),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Text(
                                        text = when {
                                            isSubTargetSegmenting -> "Ziel #${nextSubTargetId}..."
                                            subTargetMode -> "Ziele: ${overlayState.subTargets.size}"
                                            samScanResults.isNotEmpty() -> "Segmente: ${samScanResults.size}"
                                            isSamScanRunning -> "Vollscan läuft"
                                            else -> ""
                                        },
                                        color = Color.White,
                                        fontSize = 11.sp,
                                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                                    )
                                }
                            }
                            // Abbrechen-Button (oben rechts, immer sichtbar)
                            androidx.compose.material3.IconButton(
                                modifier = Modifier.align(Alignment.TopEnd).padding(8.dp).size(36.dp).background(tactDark.copy(alpha = 0.5f), RoundedCornerShape(4.dp)),
                                onClick = {
                                    subTargetMode = false
                                    clearSamScanResults()
                                    clearSubTargets()
                                    showSegmentationPicker = false
                                    cameraExpanded = false
                                    capturedSegBitmap?.recycle()
                                    capturedSegBitmap = null
                                    capturedSegFrameRef.set(null)
                                    autoDetectStateLabel = "idle"
                                    autoDetectPhase = ""
                                    statusText = "USER ABORT"
                                }
                            ) { Text("\u2715", color = tactRed, fontWeight = FontWeight.Bold) }
                        }
                    }
                }

                // Fortschritt während Aufnahme / Segmentierung
                if (isAutoDetectRunning && !showSegmentationPicker) {
                    androidx.compose.material3.Surface(
                        modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 6.dp),
                        color = Color(0xBB000000),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(
                            "⏳ ${autoDetectPhase.ifBlank { "Erkennung läuft…" }}",
                            style = MaterialTheme.typography.labelSmall,
                            color = Color(0xFFFFEA00),
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                        )
                    }
                }

                }
            }

            if (!cameraShouldShow && cameraShouldBeBound) {
                AndroidView(
                    modifier = Modifier.size(1.dp),
                    factory = { ctx ->
                        PreviewView(ctx).apply {
                            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                            scaleType = PreviewView.ScaleType.FILL_CENTER
                        }.also { previewView = it }
                    },
                    update = { previewView = it }
                )
            }

            if (!segmentationFullscreen) {
                // ── Status-Leiste (Themed) ──────────────────────────────────────────
                Box(modifier = Modifier.fillMaxWidth().background(tactDark).padding(horizontal = 12.dp, vertical = 6.dp)) {
                    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(text = statusText.uppercase(), style = MaterialTheme.typography.labelSmall, color = tactText, fontWeight = FontWeight.Bold, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
                        val samColor = when {
                            samDiagText.contains("aktiv") -> tactAccent
                            samDiagText.contains("bereit") -> tactTextDim
                            samDiagText.contains("Heuristik") -> tactAmber
                            samDiagText.contains("Fehler") -> tactRed
                            else -> tactTextDim
                        }
                        Text(
                            text = samDiagText.uppercase(),
                            style = MaterialTheme.typography.labelSmall,
                            color = samColor,
                            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                            maxLines = 1
                        )
                    }
                }

                // ── Tab-Leiste (Tactical) ──────────────────────────────────────────
                PrimaryScrollableTabRow(
                    selectedTabIndex = activeTab.ordinal,
                    edgePadding = 0.dp,
                    containerColor = tactGray,
                    contentColor = tactAccent,
                    divider = { HorizontalDivider(color = tactBorder) }
                ) {
                    AppTab.entries.forEach { tab ->
                        val isLocked = !onboardingDone && tab == AppTab.LOSLEGEN
                        Tab(
                            selected = activeTab == tab,
                            onClick = { if (!isLocked) activeTab = tab else { statusText = "INITIALIZE SYSTEM FIRST" } },
                            text = {
                                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(vertical = 8.dp)) {
                                    Text(
                                        tab.label.uppercase(),
                                        maxLines = 1,
                                        fontSize = 11.sp,
                                        fontWeight = if (activeTab == tab) FontWeight.ExtraBold else FontWeight.Bold,
                                        color = if (isLocked) tactTextDim.copy(alpha = 0.3f) else if (activeTab == tab) tactAccent else tactTextDim,
                                        fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                                        letterSpacing = 1.sp
                                    )
                                    if (isLocked) {
                                        Text("", fontSize = 8.sp, color = tactTextDim.copy(alpha = 0.3f))
                                    }
                                }
                            }
                        )
                    }
                }

                // ── Tab-Content ────────────────────────────────────────────────────
                Box(modifier = Modifier.fillMaxWidth().weight(if (cameraShouldShow) 0.58f else 1f)) {
                    Column(modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 12.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        when (activeTab) {

                        AppTab.LOSLEGEN -> {
                            if (!permissionReady) {
                                Card(modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color(0xFF3A1A1A)), border = androidx.compose.foundation.BorderStroke(1.dp, tactRed)) {
                                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                        Text("⚠ Rechte fehlen", color = tactRed, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.labelSmall)
                                        Button(modifier = Modifier.fillMaxWidth(), onClick = {
                                            permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA))
                                        }, shape = RoundedCornerShape(4.dp), colors = ButtonDefaults.buttonColors(containerColor = tactRed)) { Text("Rechte erteilen") }
                                    }
                                }
                            }
                            SectionCard(title = "LOADOUT & MATCH") {
                                OutlinedTextField(modifier = Modifier.fillMaxWidth(), value = loadoutName, onValueChange = { loadoutName = it.take(24) }, singleLine = true, label = { Text("Loadout") })
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(modifier = Modifier.weight(1f), colors = if (playerCount == 1) ButtonDefaults.buttonColors(containerColor = tactAccent, contentColor = tactDark) else ButtonDefaults.buttonColors(containerColor = tactGrayLight), onClick = { playerCount = 1; setActivePlayerSlot(1) }, shape = RoundedCornerShape(4.dp)) { Text("1 Spieler") }
                                    Button(modifier = Modifier.weight(1f), colors = if (playerCount == 2) ButtonDefaults.buttonColors(containerColor = tactAccent, contentColor = tactDark) else ButtonDefaults.buttonColors(containerColor = tactGrayLight), onClick = { playerCount = 2; if (player2Name.isBlank()) player2Name = "John Doe"; ensureWeaponAssignments() }, shape = RoundedCornerShape(4.dp)) { Text("2 Spieler") }
                                }
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    GameMode.entries.forEach { mode ->
                                        val selected = gameMode == mode
                                        Button(modifier = Modifier.weight(1f), shape = RoundedCornerShape(4.dp), colors = if (selected) ButtonDefaults.buttonColors(containerColor = tactAccent, contentColor = tactDark) else ButtonDefaults.buttonColors(containerColor = tactGrayLight), onClick = { gameMode = mode; startNewRound(); statusText = "Modus: ${gameModeLabel(mode)}" }) { Text(gameModeLabel(mode), fontSize = 9.sp, maxLines = 1, softWrap = false) }
                                    }
                                }
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(modifier = Modifier.weight(1f), colors = if (activePlayer == 1) ButtonDefaults.buttonColors(containerColor = tactAccent, contentColor = tactDark) else ButtonDefaults.buttonColors(containerColor = tactGrayLight), onClick = { setActivePlayerSlot(1) }, shape = RoundedCornerShape(4.dp)) { Text(playerName.ifBlank { "Spieler 1" }.take(14)) }
                                    Button(modifier = Modifier.weight(1f), enabled = playerCount > 1, colors = if (activePlayer == 2) ButtonDefaults.buttonColors(containerColor = tactAmber, contentColor = tactDark) else ButtonDefaults.buttonColors(containerColor = tactGrayLight), onClick = { setActivePlayerSlot(2) }, shape = RoundedCornerShape(4.dp)) { Text(player2Name.ifBlank { "John Doe" }.take(14)) }
                                }
                                MetricLine("Loadout", loadoutName.ifBlank { "Standard" }, tactAccent)
                                MetricLine("Spieler 1", "${playerName.ifBlank { "Spieler 1" }} • ${weaponNameForId(player1WeaponId)}")
                                if (playerCount > 1) MetricLine("Spieler 2", "${player2Name.ifBlank { "John Doe" }} • ${weaponNameForId(player2WeaponId)}")
                                MetricLine("Startklar", if (loadoutReady) "ja" else "Waffen noch zuweisen", if (loadoutReady) tactAccent else tactAmber)
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(modifier = Modifier.weight(1f), onClick = { activeTab = AppTab.SPIELER }, shape = RoundedCornerShape(4.dp)) { Text("Spieler") }
                                    Button(modifier = Modifier.weight(1f), onClick = { activeTab = AppTab.WAFFEN }, shape = RoundedCornerShape(4.dp), colors = ButtonDefaults.buttonColors(containerColor = tactGrayLight)) { Text("Waffen") }
                                }
                            }
                            SectionCard(title = "LOSLEGEN") {
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(
                                        modifier = Modifier.weight(1f).height(48.dp),
                                        enabled = !isRunning && permissionReady && loadoutReady,
                                        onClick = {
                                            syncRuntimePermissions()
                                            if (!hasMicPermission || !hasCameraPermission) { pendingStartAfterPermission = true; permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA)); return@Button }
                                            applyAssignedWeaponForPlayer(activePlayer)
                                            startPipeline()
                                            if (!onboardingDone && onboardingReady) { onboardingDone = true; updateOnboardingStep(); saveProfiles() }
                                        },
                                        shape = RoundedCornerShape(2.dp),
                                        elevation = ButtonDefaults.buttonElevation(defaultElevation = 6.dp, pressedElevation = 0.dp),
                                        colors = ButtonDefaults.buttonColors(containerColor = if (isRunning) tactGrayLight else tactAccent, contentColor = if (isRunning) tactTextDim else tactDark)
                                    ) { Text(if (isRunning) "LÄUFT" else "START", fontWeight = FontWeight.ExtraBold) }

                                    Button(
                                        modifier = Modifier.weight(1f).height(48.dp),
                                        enabled = isRunning,
                                        onClick = {
                                            isRunning = false; pendingStartAfterPermission = false; saveProfiles()
                                            detector.stop(); firstFrameSeen = false; firstAudioSeen = false
                                            overlayState = overlayState.copy(hitWasValid = null)
                                            statusText = "Betrieb gestoppt"
                                        },
                                        colors = ButtonDefaults.buttonColors(containerColor = tactGrayLight, contentColor = tactText),
                                        shape = RoundedCornerShape(2.dp),
                                        elevation = ButtonDefaults.buttonElevation(defaultElevation = 2.dp)
                                    ) { Text("STOP", fontWeight = FontWeight.Bold) }
                                }
                                if (gameMode == GameMode.DRILL && drillAwaitingWeaponSwitch) {
                                    Button(modifier = Modifier.fillMaxWidth(), colors = ButtonDefaults.buttonColors(containerColor = tactRed), onClick = {
                                        drillAwaitingWeaponSwitch = false; speak("Weiter")
                                    }, shape = RoundedCornerShape(4.dp)) { Text("Waffenwechsel bestätigen") }
                                }
                            }
                            SectionCard(title = "MATCH-STATUS") {
                                Row(Modifier.fillMaxWidth()) {
                                    Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                                        Text(playerName.uppercase(), style = MaterialTheme.typography.labelMedium, color = tactAccent, fontWeight = FontWeight.ExtraBold)
                                        Text("$scoreP1", style = MaterialTheme.typography.displaySmall, fontWeight = FontWeight.Bold, color = tactAccent, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
                                        Text("${hitsP1} Treffer", style = MaterialTheme.typography.labelSmall, color = tactTextDim)
                                    }
                                    if (playerCount > 1) {
                                        Box(Modifier.width(1.dp).height(80.dp).background(tactGrayLight))
                                        Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                                            Text(player2Name.uppercase(), style = MaterialTheme.typography.labelMedium, color = tactAmber, fontWeight = FontWeight.ExtraBold)
                                            Text("$scoreP2", style = MaterialTheme.typography.displaySmall, fontWeight = FontWeight.Bold, color = tactAmber, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
                                            Text("${hitsP2} Treffer", style = MaterialTheme.typography.labelSmall, color = tactTextDim)
                                        }
                                    }
                                }
                                HorizontalDivider(color = tactGrayLight)
                                MetricLine("Modus", gameModeLabel(gameMode), tactAccent)
                                MetricLine("Aktiver Spieler", if (activePlayer == 1) playerName.ifBlank { "Spieler 1" } else player2Name.ifBlank { "John Doe" }, if (activePlayer == 1) tactAccent else tactAmber)
                                MetricLine("Aktive Waffe", weaponName, tactText)
                                if (gameMode == GameMode.DRILL) MetricLine("Drill", "$drillZone ×$drillRemaining", tactAmber)
                                MetricLine("Letztes Event", hitValidationInfo, if (hitValidationInfo.contains("Treffer")) tactAccent else tactTextDim)
                            }
                            SectionCard(title = "SIGNAL-LOG") {
                                if (shotHistory.isEmpty()) {
                                    Text("Noch keine Daten", style = MaterialTheme.typography.labelSmall, color = tactTextDim, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
                                } else {
                                    shotHistory.take(15).forEach { rec ->
                                        val isHit = rec.awarded > 0
                                        val color = if (isHit) tactAccent else tactRed
                                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                            Text("> ${rec.playerName.uppercase()}", color = if (rec.playerName == playerName) tactAccent else tactAmber, style = MaterialTheme.typography.labelSmall, modifier = Modifier.width(80.dp))
                                            Text(if (isHit) "HIT" else "MISS", color = color, style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, modifier = Modifier.width(40.dp))
                                            Text(rec.reason.uppercase(), style = MaterialTheme.typography.labelSmall, color = tactTextDim, modifier = Modifier.weight(1f), maxLines = 1)
                                        }
                                    }
                                }
                            }
                            if (!onboardingReady || !loadoutReady) {
                                Card(modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1E1E))) {
                                    Row(Modifier.padding(12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                                        Text("Hinweis:", color = Color(0xFFFF8A80), fontWeight = FontWeight.Bold)
                                        Text(if (!loadoutReady) "Weise im Tab 'Spieler' oder 'Waffen' jedem aktiven Slot eine Waffe zu." else onboardingStepInstruction(onboardingStep), style = MaterialTheme.typography.bodySmall, color = Color(0xFFBDBDBD))
                                    }
                                }
                            }
                        }

                        AppTab.SPIELER -> {
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                                SectionCard(title = "SPIELER 1", modifier = Modifier.weight(1f)) {
                                    OutlinedTextField(modifier = Modifier.fillMaxWidth(), value = playerName, onValueChange = { playerName = it.take(16) }, singleLine = true, label = { Text("Name", style = MaterialTheme.typography.labelSmall) }, textStyle = androidx.compose.ui.text.TextStyle(fontWeight = FontWeight.Bold, fontSize = 16.sp))
                                    MetricLine("Waffe", weaponNameForId(player1WeaponId), tactAccent)
                                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                        Button(modifier = Modifier.weight(1f), onClick = { assignSelectedWeaponToPlayer(1); saveProfiles() }, shape = RoundedCornerShape(4.dp)) { Text("Aktive Waffe") }
                                        Button(modifier = Modifier.weight(1f), onClick = { setActivePlayerSlot(1); activeTab = AppTab.WAFFEN }, colors = ButtonDefaults.buttonColors(containerColor = tactGrayLight), shape = RoundedCornerShape(4.dp)) { Text("Waffen") }
                                    }
                                }
                                SectionCard(title = if (playerCount > 1) "SPIELER 2 / GAST" else "GASTSLOT", modifier = Modifier.weight(1f)) {
                                    OutlinedTextField(modifier = Modifier.fillMaxWidth(), value = player2Name, onValueChange = { player2Name = it.take(16) }, singleLine = true, label = { Text("Name", style = MaterialTheme.typography.labelSmall) }, textStyle = androidx.compose.ui.text.TextStyle(fontWeight = FontWeight.Bold, fontSize = 16.sp))
                                    MetricLine("Waffe", if (playerCount > 1) weaponNameForId(player2WeaponId) else "nicht aktiv", tactAmber)
                                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                        Button(modifier = Modifier.weight(1f), enabled = playerCount > 1, onClick = { assignSelectedWeaponToPlayer(2); saveProfiles() }, shape = RoundedCornerShape(4.dp)) { Text("Aktive Waffe") }
                                        Button(modifier = Modifier.weight(1f), onClick = { playerCount = 2; player2Name = "John Doe"; saveProfiles(); statusText = "Gastspieler vorbereitet" }, colors = ButtonDefaults.buttonColors(containerColor = tactGrayLight), shape = RoundedCornerShape(4.dp)) { Text("John Doe") }
                                    }
                                }
                            }
                            SectionCard(title = "LOADOUT-ZUORDNUNG") {
                                MetricLine("Loadout", loadoutName.ifBlank { "Standard" }, tactAccent)
                                MetricLine("Spieleranzahl", playerCount.toString())
                                MetricLine("Spieler 1", "${playerName.ifBlank { "Spieler 1" }} • ${weaponNameForId(player1WeaponId)}")
                                MetricLine("Spieler 2", if (playerCount > 1) "${player2Name.ifBlank { "John Doe" }} • ${weaponNameForId(player2WeaponId)}" else "nicht aktiv")
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(modifier = Modifier.weight(1f), onClick = { activeTab = AppTab.WAFFEN }, shape = RoundedCornerShape(4.dp)) { Text("Waffen verwalten") }
                                    Button(modifier = Modifier.weight(1f), onClick = { activeTab = AppTab.LOSLEGEN }, colors = ButtonDefaults.buttonColors(containerColor = tactGrayLight), shape = RoundedCornerShape(4.dp)) { Text("Zu Loslegen") }
                                }
                            }
                            Button(modifier = Modifier.fillMaxWidth().height(56.dp), onClick = { saveProfiles(); updateOnboardingStep(); statusText = "Spieler gespeichert" }, shape = RoundedCornerShape(4.dp), colors = ButtonDefaults.buttonColors(containerColor = tactAccent, contentColor = tactDark)) { Text("SPIELER SPEICHERN", fontWeight = FontWeight.Bold) }
                        }

                        AppTab.WAFFEN -> {
                            SectionCard(title = "WAFFENPROFILE") {
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                                    OutlinedTextField(modifier = Modifier.weight(1f), value = weaponName, onValueChange = { weaponName = it.take(24) }, singleLine = true, label = { Text("Model") })
                                    Button(onClick = { photoPickerLauncher.launch("image/*") }, colors = ButtonDefaults.buttonColors(containerColor = tactGrayLight), shape = RoundedCornerShape(4.dp)) { Text("Foto", fontSize = 10.sp) }
                                }
                                if (weaponPhotoUri.isNotBlank()) {
                                    val bmp = remember(weaponPhotoUri) { decodeSampledBitmapFromUri(context, weaponPhotoUri) }
                                    if (bmp != null) {
                                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                            Image(bitmap = bmp.asImageBitmap(), contentDescription = "Waffenfoto", modifier = Modifier.size(80.dp).clip(RoundedCornerShape(4.dp)).background(tactGray), contentScale = ContentScale.Crop)
                                            Column {
                                                Text(weaponName.ifBlank { "Unbekannt" }, fontWeight = FontWeight.ExtraBold, fontSize = 18.sp, color = tactAccent)
                                                Text("ID: ${selectedWeaponId.take(8)}", style = MaterialTheme.typography.labelSmall, color = tactTextDim, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
                                            }
                                        }
                                    }
                                }
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    OutlinedTextField(modifier = Modifier.weight(1f), value = weaponFpsInput, onValueChange = { weaponFpsInput = it.take(4) }, singleLine = true, label = { Text("FPS") })
                                    OutlinedTextField(modifier = Modifier.weight(1f), value = weaponBbWeightInput, onValueChange = { weaponBbWeightInput = it.take(6) }, singleLine = true, label = { Text("BB (g)") })
                                }
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(modifier = Modifier.weight(1f), onClick = { upsertCurrentWeaponProfile(); saveProfiles(); statusText = "Waffenprofil gespeichert" }, shape = RoundedCornerShape(4.dp)) { Text("Speichern") }
                                    Button(modifier = Modifier.weight(1f), enabled = weaponProfiles.size > 1, onClick = {
                                        val idx = weaponProfiles.indexOfFirst { it.id == selectedWeaponId }
                                        val next = if (idx < 0) 0 else (idx + 1) % weaponProfiles.size
                                        selectedWeaponId = weaponProfiles[next].id; applySelectedWeaponProfile()
                                    }, colors = ButtonDefaults.buttonColors(containerColor = tactGrayLight), shape = RoundedCornerShape(4.dp)) { Text("Nächstes Profil") }
                                }
                            }
                            SectionCard(title = "ZUWEISUNG") {
                                MetricLine("Aktuelles Profil", weaponName.ifBlank { "Waffe" }, tactAccent)
                                MetricLine("Spieler 1 nutzt", weaponNameForId(player1WeaponId))
                                MetricLine("Spieler 2 nutzt", if (playerCount > 1) weaponNameForId(player2WeaponId) else "nicht aktiv")
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(modifier = Modifier.weight(1f), onClick = { assignSelectedWeaponToPlayer(1); saveProfiles() }, shape = RoundedCornerShape(4.dp)) { Text("→ Spieler 1") }
                                    Button(modifier = Modifier.weight(1f), enabled = playerCount > 1, onClick = { assignSelectedWeaponToPlayer(2); saveProfiles() }, shape = RoundedCornerShape(4.dp), colors = ButtonDefaults.buttonColors(containerColor = tactAmber, contentColor = tactDark)) { Text("→ Spieler 2") }
                                }
                                Button(modifier = Modifier.fillMaxWidth(), onClick = { activeTab = AppTab.SPIELER }, colors = ButtonDefaults.buttonColors(containerColor = tactGrayLight), shape = RoundedCornerShape(4.dp)) { Text("Spieler-Zuordnung ansehen") }
                            }
                            Button(modifier = Modifier.fillMaxWidth().height(56.dp), onClick = { saveProfiles(); updateOnboardingStep(); statusText = "Waffen gespeichert" }, shape = RoundedCornerShape(4.dp), colors = ButtonDefaults.buttonColors(containerColor = tactAccent, contentColor = tactDark)) { Text("WAFFEN SPEICHERN", fontWeight = FontWeight.Bold) }
                        }

                        AppTab.SETUP -> {
                            if (!permissionReady) {
                                Card(modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color(0xFF3A1A1A)), border = androidx.compose.foundation.BorderStroke(1.dp, tactRed)) {
                                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                        Text("⚠ Rechte fehlen", color = tactRed, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.labelSmall)
                                        Text("Mikrofon und Kamera werden für Zielerfassung und Kalibrierung benötigt.", color = tactText)
                                        Button(modifier = Modifier.fillMaxWidth(), onClick = {
                                            permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA))
                                        }, shape = RoundedCornerShape(4.dp), colors = ButtonDefaults.buttonColors(containerColor = tactRed)) { Text("Rechte erteilen") }
                                    }
                                }
                            }
                            SectionCard(title = "SETUP-STATUS") {
                                MetricLine("Spieler/Waffen", if (loadoutReady) "zugewiesen" else "noch offen", if (loadoutReady) tactAccent else tactAmber)
                                MetricLine("Rechte", if (permissionReady) "ok" else "fehlen", if (permissionReady) tactAccent else tactRed)
                                MetricLine("Ziel", if (boardReady) "erfasst" else "offen", if (boardReady) tactAccent else tactAmber)
                                MetricLine("Sound", if (calibrationReady) "kalibriert" else "offen", if (calibrationReady) tactAccent else tactAmber)
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(modifier = Modifier.weight(1f), onClick = { activeTab = AppTab.SPIELER }, shape = RoundedCornerShape(4.dp)) { Text("Spieler") }
                                    Button(modifier = Modifier.weight(1f), onClick = { activeTab = AppTab.WAFFEN }, colors = ButtonDefaults.buttonColors(containerColor = tactGrayLight), shape = RoundedCornerShape(4.dp)) { Text("Waffen") }
                                }
                            }
                            SectionCard(title = "Zielplatte (Board)") {
                                 Text("Kamera auf Zielplatte richten und 'Auto-Erkennen' tippen. Foto wird aufgenommen, dann Ziel antippen.", style = MaterialTheme.typography.bodySmall, color = Color(0xFFBBBBBB))
                                 Button(modifier = Modifier.fillMaxWidth(), enabled = hasCameraPermission && !isAutoDetectRunning, onClick = { startSegCapture() }) {
                                     Text(if (isAutoDetectRunning) "Erkennung l\u00E4uft..." else "Auto-Erkennen")
                                 }
                                if (isAutoDetectRunning) {
                                    LinearProgressIndicator(progress = { autoDetectProgress.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                                    Text(autoDetectPhase, style = MaterialTheme.typography.labelSmall, color = Color(0xFFFF8A80))
                                }
                                if (boardReady) {
                                    Button(modifier = Modifier.fillMaxWidth().height(48.dp), colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1B5E20)), onClick = { updateOnboardingStep() }) { Text("\u2714 Ziel bestätigt") }
                                }
                                MetricLine("Status", if (boardReady) "\u2714 Platte gesetzt" else "Nicht gesetzt")
                                MetricLine("Auto-Detect", "$autoDetectStateLabel (${autoDetectRuntimeMs}ms)")
                            }
                            SectionCard(title = "Schuss-Kalibrierung") {
                                Text("5 saubere Sch\u00FCsse schie\u00DFen \u2013 die App z\u00E4hlt automatisch.", style = MaterialTheme.typography.bodySmall, color = Color(0xFFBBBBBB))
                                if (isFiveShotCalibrationActive) {
                                    LinearProgressIndicator(progress = { calibrationShots / 5f }, modifier = Modifier.fillMaxWidth())
                                    Text("$calibrationShots / 5 Sch\u00FCsse erkannt", color = tactAmber, fontWeight = FontWeight.Bold, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
                                }
                                Button(modifier = Modifier.fillMaxWidth(), enabled = !isFiveShotCalibrationActive, onClick = { beginFiveShotCalibration() }) {
                                    Text(if (calibrationReady) "\u2714 Kalibriert \u2013 neu starten" else "Kalibrierung starten")
                                }
                                if (calibrationReady) {
                                    Button(modifier = Modifier.fillMaxWidth().height(56.dp), colors = ButtonDefaults.buttonColors(containerColor = tactAccent, contentColor = tactDark), onClick = { onboardingDone = true; setupLayerDismissed = true; saveProfiles(); statusText = "SYSTEM ONLINE" }, shape = RoundedCornerShape(2.dp)) { Text("SYSTEM ONLINE - ENGAGE", fontWeight = FontWeight.ExtraBold) }
                                }
                            }
                        }

                        AppTab.SYSTEM -> {
                            SectionCard(title = "HARDWARE INTERFACE") {
                                MetricLine("Optical Unit", if (cameraReady) "ACTIVE" else "OFFLINE", if (cameraReady) tactAccent else tactRed)
                                MetricLine("Acoustic Unit", if (audioReady) "ACTIVE" else "OFFLINE", if (audioReady) tactAccent else tactRed)
                                MetricLine("Frame Buff", "$frameTick", tactTextDim)
                                MetricLine("Signal Buff", "$audioTick", tactTextDim)
                                MetricLine("Sichtbare Kamera", if (cameraShouldShow) "an" else "aus")
                                MetricLine("Kamera gebunden", if (cameraShouldBeBound) "ja" else "nein")
                            }
                            SectionCard(title = "ERKENNUNGS-SCHWELLEN") {
                                Text("Peak/Noise: ${String.format(Locale.US, "%.1f", configPeakThresh)}", style = MaterialTheme.typography.bodySmall)
                                Slider(value = configPeakThresh, onValueChange = { configPeakThresh = it }, valueRange = 1.5f..8.0f, steps = 26, modifier = Modifier.fillMaxWidth())
                                Text("Cooldown: ${configCooldownMs.toInt()} ms", style = MaterialTheme.typography.bodySmall)
                                Slider(value = configCooldownMs, onValueChange = { configCooldownMs = it }, valueRange = 80f..800f, steps = 35, modifier = Modifier.fillMaxWidth())
                                Text("Bewegungs-Diff: $configDiffThreshold", style = MaterialTheme.typography.bodySmall)
                                Slider(value = configDiffThreshold.toFloat(), onValueChange = { configDiffThreshold = it.toInt() }, valueRange = 5f..60f, steps = 55, modifier = Modifier.fillMaxWidth())
                            }
                        }


                        AppTab.DETAILS -> {
                            SectionCard(title = "Telemetrie") {
                                MetricLine("Schuss (Audio)", shotCount.toString())
                                MetricLine("Trigger", triggerCount.toString())
                                MetricLine("Letzter Schuss", lastShotAt)
                                MetricLine("Letztes Audio", lastAudioAt)
                                MetricLine("Noise Floor", format3(detectorState.noiseFloor))
                                MetricLine("Peak/Noise", format3(detectorState.peakToNoise))
                                MetricLine("Motion Score", format3(motionState.score))
                                MetricLine("YOLO-Conf", lastYolo)
                                MetricLine("Detektor", lastDetectorSource)
                                MetricLine("Seg-Backend", lastSegmentationBackend)
                                MetricLine("Auto-Detect", lastAutoDetect)
                                MetricLine("Detect-Error", autoDetectErrorCode)
                                MetricLine("Trajectory", lastTrajectory)
                                MetricLine("Treffer", lastHit)
                                MetricLine("Audio-Mux", lastAudioMux)
                                MetricLine("Export", lastExport.take(60))
                            }
                            if (replayVideoPath != null) {
                                Button(modifier = Modifier.fillMaxWidth(), onClick = { showReplayViewer = true }) { Text("\u25B6  Letzten Replay ansehen") }
                            }
                        }
                    }
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                }
            }
        } // end Column

        // ── Setup-Overlay ─────────────────────────────────────────────────────
        if (setupLayerActive && !segmentationFullscreen) {
            val completedSteps = (if (profileReady) 1 else 0) + (if (permissionReady) 1 else 0) + (if (boardReady) 1 else 0) + (if (calibrationReady) 1 else 0)
            Box(modifier = Modifier.fillMaxSize()) {
                Box(modifier = Modifier.fillMaxWidth().fillMaxHeight(0.44f).background(Color(0x22000000)))
                Box(modifier = Modifier.fillMaxWidth().align(Alignment.BottomCenter).fillMaxHeight(0.65f).background(tactDark).then(Modifier.background(tactBorder))) {
                    Column(modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Text("APP-SETUP", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.ExtraBold, color = tactAccent, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace, letterSpacing = 2.sp)
                            Spacer(modifier = Modifier.weight(1f))
                            Text("${completedSteps + 1}/4", style = MaterialTheme.typography.labelSmall, color = tactAmber, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
                        }
                        
                        Row(Modifier.fillMaxWidth().height(4.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            repeat(4) { i ->
                                Box(Modifier.weight(1f).fillMaxHeight().background(if (i <= completedSteps) tactAccent else tactGrayLight))
                            }
                        }

                        val setupSteps = listOf(
                            OnboardingStep.PROFILE to "SPIELER & WAFFEN", 
                            OnboardingStep.PERMISSIONS to "RECHTE", 
                            OnboardingStep.BOARD to "ZIEL", 
                            OnboardingStep.CALIBRATION to "SOUND"
                        )
                        setupSteps.forEachIndexed { idx, (step, label) ->
                            val isDone = when (step) { OnboardingStep.PROFILE -> profileReady; OnboardingStep.PERMISSIONS -> permissionReady; OnboardingStep.BOARD -> boardReady; OnboardingStep.CALIBRATION -> calibrationReady; else -> false }
                            val isCurrent = onboardingStep == step
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                                Text(if (isDone) "\u2714" else "${idx + 1}", color = if (isDone) Color(0xFF66BB6A) else if (isCurrent) Color(0xFFFF5252) else Color(0xFF757575), fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
                                Text(label, color = if (isDone) Color(0xFF66BB6A) else if (isCurrent) Color(0xFFFF5252) else Color(0xFF757575), fontWeight = if (isCurrent) FontWeight.Bold else FontWeight.Normal)
                            }
                        }
                        HorizontalDivider(color = Color(0xFF333333))
                        when (onboardingStep) {
                            OnboardingStep.PROFILE -> {
                                SectionCard(title = "SPIELER & WAFFEN") {
                                    OutlinedTextField(modifier = Modifier.fillMaxWidth(), value = playerName, onValueChange = { playerName = it.take(24) }, singleLine = true, label = { Text("PLAYER 1 CALLSIGN") })
                                    OutlinedTextField(modifier = Modifier.fillMaxWidth(), value = player2Name, onValueChange = { player2Name = it.take(24) }, singleLine = true, label = { Text("PLAYER 2 CALLSIGN") })
                                    MetricLine("Loadout", loadoutName.ifBlank { "Standard" }, tactAccent)
                                    MetricLine("Spieler 1 Waffe", weaponNameForId(player1WeaponId))
                                    MetricLine("Spieler 2 Waffe", if (playerCount > 1) weaponNameForId(player2WeaponId) else "nicht aktiv")
                                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                        Button(modifier = Modifier.weight(1f), onClick = { activeTab = AppTab.SPIELER; setupLayerDismissed = true }, shape = RoundedCornerShape(2.dp)) { Text("Spieler öffnen", fontWeight = FontWeight.Bold) }
                                        Button(modifier = Modifier.weight(1f), onClick = { activeTab = AppTab.WAFFEN; setupLayerDismissed = true }, shape = RoundedCornerShape(2.dp), colors = ButtonDefaults.buttonColors(containerColor = tactGrayLight)) { Text("Waffen öffnen", fontWeight = FontWeight.Bold) }
                                    }
                                    Button(modifier = Modifier.fillMaxWidth().height(48.dp), enabled = playerName.isNotBlank() && weaponName.isNotBlank(), onClick = { saveProfiles(); updateOnboardingStep(); statusText = "Spieler gespeichert" }, shape = RoundedCornerShape(2.dp)) { Text("SPIELER SPEICHERN", fontWeight = FontWeight.Bold) }
                                }
                            }
                            OnboardingStep.PERMISSIONS -> {
                                SectionCard(title = "RECHTE") {
                                    Text("Mikrofon und Kamera werden für Zielerfassung und Sound-Kalibrierung benötigt.", style = MaterialTheme.typography.labelSmall, color = tactAmber)
                                    Button(modifier = Modifier.fillMaxWidth().height(48.dp), onClick = { permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA)) }, shape = RoundedCornerShape(2.dp)) { Text("RECHTE ERTEILEN", fontWeight = FontWeight.Bold) }
                                }
                            }
                            OnboardingStep.BOARD -> {
                                SectionCard(title = "ZIEL ERFASSEN") {
                                    Text("Richte die Kamera auf die Zielplatte und starte die Erfassung.", style = MaterialTheme.typography.labelSmall, color = tactTextDim)
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Button(modifier = Modifier.fillMaxWidth().height(48.dp), enabled = hasCameraPermission && !isAutoDetectRunning, onClick = { startSegCapture() }, shape = RoundedCornerShape(2.dp)) {
                                        Text(if (isAutoDetectRunning) "ERFASSE..." else "ZIELERFASSUNG STARTEN", fontWeight = FontWeight.Bold)
                                    }
                                    if (isAutoDetectRunning) {
                                        LinearProgressIndicator(progress = { autoDetectProgress.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth(), color = tactAccent, trackColor = tactGrayLight)
                                    }
                                    if (boardReady) {
                                        Button(modifier = Modifier.fillMaxWidth().height(48.dp), colors = ButtonDefaults.buttonColors(containerColor = tactAccent, contentColor = tactDark), onClick = { updateOnboardingStep() }, shape = RoundedCornerShape(2.dp)) { Text("ZIEL BESTÄTIGEN", fontWeight = FontWeight.ExtraBold) }
                                    }
                                }
                            }
                            OnboardingStep.CALIBRATION -> {
                                SectionCard(title = "SOUND-KALIBRIERUNG") {
                                    Text("Gib 5 saubere Schüsse ab, damit die Audioerkennung sauber auf die Waffe eingelernt wird.", style = MaterialTheme.typography.labelSmall, color = tactTextDim)
                                    if (isFiveShotCalibrationActive) {
                                        LinearProgressIndicator(progress = { calibrationShots / 5f }, modifier = Modifier.fillMaxWidth(), color = tactAmber, trackColor = tactGrayLight)
                                        Text("ERKANNT: $calibrationShots / 5", color = tactAmber, fontWeight = FontWeight.Bold, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
                                    }
                                    Button(modifier = Modifier.fillMaxWidth().height(48.dp), enabled = !isFiveShotCalibrationActive, onClick = { beginFiveShotCalibration() }, shape = RoundedCornerShape(2.dp)) {
                                        Text(if (calibrationReady) "NEU KALIBRIEREN" else "KALIBRIERUNG STARTEN", fontWeight = FontWeight.Bold)
                                    }
                                    if (calibrationReady) {
                                        Button(modifier = Modifier.fillMaxWidth().height(56.dp), colors = ButtonDefaults.buttonColors(containerColor = tactAccent, contentColor = tactDark), onClick = { onboardingDone = true; setupLayerDismissed = true; saveProfiles(); statusText = "SYSTEM ONLINE" }, shape = RoundedCornerShape(2.dp)) { Text("SETUP ABSCHLIESSEN", fontWeight = FontWeight.ExtraBold) }
                                    }
                                }
                            }
                            OnboardingStep.READY -> {
                                Button(modifier = Modifier.fillMaxWidth().height(56.dp), colors = ButtonDefaults.buttonColors(containerColor = tactAccent, contentColor = tactDark), onClick = { onboardingDone = true; setupLayerDismissed = true; saveProfiles() }, shape = RoundedCornerShape(2.dp)) { Text("LOSLEGEN", fontWeight = FontWeight.ExtraBold) }
                            }
                        }
                        Button(modifier = Modifier.fillMaxWidth(), colors = ButtonDefaults.buttonColors(containerColor = Color.Transparent, contentColor = tactTextDim), onClick = { setupLayerDismissed = true; statusText = "SETUP SPÄTER FORTSETZEN" }) { Text("SPÄTER FORTSETZEN", style = MaterialTheme.typography.labelSmall) }
                    }
                }
            }
        }

        // ── Replay Viewer ─────────────────────────────────────────────────
        if (showReplayViewer && replayVideoPath != null) {
            Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
                val videoFile = remember(replayVideoPath!!) { File(replayVideoPath!!) }
                if (videoFile.exists()) {
                    AndroidView(modifier = Modifier.fillMaxSize(), factory = { ctx ->
                        VideoView(ctx).apply {
                          val mc = MediaController(ctx); mc.setAnchorView(this); setMediaController(mc)
                          setVideoURI(Uri.fromFile(videoFile)); requestFocus()
                          setOnPreparedListener { it.isLooping = false; it.start() }
                        }
                    })
                } else {
                    Text(text = "Videodatei nicht gefunden:\n${replayVideoPath}", color = Color.White, modifier = Modifier.align(Alignment.Center).padding(24.dp))
                }
                Button(onClick = { showReplayViewer = false }, modifier = Modifier.align(Alignment.TopEnd).padding(16.dp)) { Text("\u2715  Schliessen") }
            }
        }

    } // end Box

} // end AirsoftPipelineScreen

@Composable
private fun MetricLine(label: String, value: String, valueColor: Color = Color.Unspecified) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = Color(0xFF9E9E9E))
        Text(value, fontWeight = FontWeight.Medium, color = valueColor, style = MaterialTheme.typography.labelMedium, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace)
    }
}

@Composable
private fun SectionCard(title: String, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF14141A)),
        shape = RoundedCornerShape(2.dp), // Sharper corners for tactical look
        border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFF2D2D3A)),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp) // Real depth
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                title.uppercase(),
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.ExtraBold,
                color = Color(0xFF00E676),
                fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                letterSpacing = 1.2.sp
            )
            content()
        }
    }
}

private fun format3(value: Double): String = String.format(Locale.US, "%.3f", value)


