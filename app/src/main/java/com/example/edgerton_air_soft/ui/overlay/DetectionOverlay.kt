package com.example.edgerton_air_soft.ui.overlay

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.example.edgerton_air_soft.segmentation.TargetEntry

data class OverlayViewState(
    val boardTlXNorm: Double? = null,
    val boardTlYNorm: Double? = null,
    val boardTrXNorm: Double? = null,
    val boardTrYNorm: Double? = null,
    val boardBrXNorm: Double? = null,
    val boardBrYNorm: Double? = null,
    val boardBlXNorm: Double? = null,
    val boardBlYNorm: Double? = null,
    // Pixel-Maske von SAM (nur wenn Modell geladen + Inferenz erfolgreich)
    val maskBitmap: Bitmap? = null,
    // Erkannte Sub-Ziele auf der Platte (normalisierte Kamera-Koordinaten)
    val detectedTargets: List<Pair<Double, Double>> = emptyList(),
    // Stage-2-Sub-Ziele mit individueller Pixel-Maske (Multi-Tap-Workflow).
    // Wenn leer, wird das alte Verhalten beibehalten.
    val subTargets: List<TargetEntry> = emptyList(),
    val samScanResults: List<Bitmap> = emptyList(),
    val hitXNorm: Double? = null,
    val hitYNorm: Double? = null,
    val hitWasValid: Boolean? = null,
    val targetXNorm: Double = 0.5,
    val targetYNorm: Double = 0.5
)

@Composable
fun DetectionOverlay(
    state: OverlayViewState,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        // ── Board-Ecken berechnen ─────────────────────────────────────
        fun normOff(xn: Double?, yn: Double?) =
            if (xn != null && yn != null)
                androidx.compose.ui.geometry.Offset(
                    (xn.coerceIn(0.0, 1.0) * size.width).toFloat(),
                    (yn.coerceIn(0.0, 1.0) * size.height).toFloat()
                )
            else null

        val tl = normOff(state.boardTlXNorm, state.boardTlYNorm)
        val tr = normOff(state.boardTrXNorm, state.boardTrYNorm)
        val br = normOff(state.boardBrXNorm, state.boardBrYNorm)
        val bl = normOff(state.boardBlXNorm, state.boardBlYNorm)

        // ── 1a) SAM Pixel-Maske (aktiv) ──────────────────────────────────
        state.maskBitmap?.let { mask ->
            try {
                drawImage(
                    mask.asImageBitmap(),
                    dstSize = androidx.compose.ui.unit.IntSize(size.width.toInt(), size.height.toInt()),
                    alpha = 0.82f
                )
            } catch (_: Throwable) {}
        }

        // ── 1b) SAM Scan-Ergebnisse (Vorschau zum Auswaehlen) ─────────────
        state.samScanResults.forEachIndexed { idx, mask ->
            val tint = SUB_TARGET_PALETTE[idx % SUB_TARGET_PALETTE.size]
            try {
                drawImage(
                    mask.asImageBitmap(),
                    dstSize = androidx.compose.ui.unit.IntSize(size.width.toInt(), size.height.toInt()),
                    alpha = 0.82f,
                    colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(tint)
                )
                drawImage(
                    mask.asImageBitmap(),
                    dstSize = androidx.compose.ui.unit.IntSize(size.width.toInt(), size.height.toInt()),
                    alpha = 0.35f
                )
            } catch (_: Throwable) {}
        }

        // ── 2) Board-Polygon (nur wenn keine Maske) ──────────────────
        if (state.maskBitmap == null) {
            if (tl != null && tr != null && br != null && bl != null) {
                val boardColor = Color(0xFFB388FF)
                val sw = 2.dp.toPx()
                drawLine(boardColor, tl, tr, strokeWidth = sw)
                drawLine(boardColor, tr, br, strokeWidth = sw)
                drawLine(boardColor, br, bl, strokeWidth = sw)
                drawLine(boardColor, bl, tl, strokeWidth = sw)
            }
        }

        // ── 3) Eck-Punkte für Drag-Anpassung (nur ohne Maske) ──────────
        val hasMask = state.maskBitmap != null
        if (!hasMask) {
            val dotColor = Color(0xFF00E5FF)
            val dotR = size.minDimension * 0.018f
            if (tl != null) drawCircle(dotColor, dotR, tl)
            if (tr != null) drawCircle(dotColor, dotR, tr)
            if (br != null) drawCircle(dotColor, dotR, br)
            if (bl != null) drawCircle(dotColor, dotR, bl)
        }

        // ── 4a) Stage-2-Sub-Target-Masken (eine Farbe pro Ziel) ───────
        // Pro Ziel-ID eine eigene Farbe, damit die User die Ziele in der
        // Liste unten zuordnen koennen.
        state.subTargets.forEachIndexed { idx, entry ->
            val mask = entry.maskBitmap ?: return@forEachIndexed
            val tint = SUB_TARGET_PALETTE[idx % SUB_TARGET_PALETTE.size]
            try {
                drawImage(
                    mask.asImageBitmap(),
                    dstSize = androidx.compose.ui.unit.IntSize(size.width.toInt(), size.height.toInt()),
                    alpha = 0.78f,
                    colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(tint)
                )
            } catch (_: Throwable) {}
            // Ziel-Mittelpunkt + Nummer
            val cx = (entry.centerXNorm.coerceIn(0.0, 1.0) * size.width).toFloat()
            val cy = (entry.centerYNorm.coerceIn(0.0, 1.0) * size.height).toFloat()
            drawCircle(tint, 4.dp.toPx(), androidx.compose.ui.geometry.Offset(cx, cy))
        }

        // ── 4b) Legacy: Erkannte Sub-Ziele (türkise Zielkreise) ──────
        state.detectedTargets.forEach { (tx, ty) ->
            val px = (tx.coerceIn(0.0, 1.0) * size.width).toFloat()
            val py = (ty.coerceIn(0.0, 1.0) * size.height).toFloat()
            val r  = size.minDimension * 0.045f
            drawCircle(Color(0xFF00BCD4), r, androidx.compose.ui.geometry.Offset(px, py), style = Stroke(width = 2.5.dp.toPx()))
            drawCircle(Color(0xFF00BCD4), 3.dp.toPx(), androidx.compose.ui.geometry.Offset(px, py))
        }

        // ── 5) Ziel-Crosshair ────────────────────────────────────────
        val targetPx = (state.targetXNorm.coerceIn(0.0, 1.0) * size.width).toFloat()
        val targetPy = (state.targetYNorm.coerceIn(0.0, 1.0) * size.height).toFloat()
        val targetRadius = size.minDimension * 0.06f
        drawCircle(
            color = Color(0x80FFFFFF),
            radius = targetRadius,
            center = androidx.compose.ui.geometry.Offset(targetPx, targetPy),
            style = Stroke(width = 2.dp.toPx())
        )

        // ── 6) Treffer-Marker ─────────────────────────────────────────
        val hitX = state.hitXNorm; val hitY = state.hitYNorm
        if (hitX != null && hitY != null) {
            val px = (hitX.coerceIn(0.0, 1.0) * size.width).toFloat()
            val py = (hitY.coerceIn(0.0, 1.0) * size.height).toFloat()
            val m  = size.minDimension * 0.035f
            val hitColor = when (state.hitWasValid) {
                true  -> Color(0xFF00E676)
                false -> Color(0xFFFF5252)
                null  -> Color(0xFFFFEA00)
            }
            drawCircle(hitColor, m, androidx.compose.ui.geometry.Offset(px, py), style = Stroke(width = 3.dp.toPx()))
            drawLine(hitColor, androidx.compose.ui.geometry.Offset(px - m * 1.4f, py), androidx.compose.ui.geometry.Offset(px + m * 1.4f, py), 2.dp.toPx())
            drawLine(hitColor, androidx.compose.ui.geometry.Offset(px, py - m * 1.4f), androidx.compose.ui.geometry.Offset(px, py + m * 1.4f), 2.dp.toPx())
        }
    }
}

/** Farbpalette fuer Sub-Target-Masken — durchnummeriert, damit User Ziel-Index
 *  zur Farbe in der UI-Liste zuordnen kann. */
private val SUB_TARGET_PALETTE = listOf(
    Color(0xFF00BCD4), // 1 türkis
    Color(0xFFFFAB00), // 2 amber
    Color(0xFFFF4081), // 3 pink
    Color(0xFFB388FF), // 4 lila
    Color(0xFF69F0AE), // 5 grün
    Color(0xFFFF5252), // 6 rot
    Color(0xFFFFD740), // 7 gelb
    Color(0xFF40C4FF)  // 8 himmelblau
)
