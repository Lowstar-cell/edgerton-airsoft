package com.example.edgerton_air_soft.segmentation

import android.graphics.Bitmap
import android.graphics.Color
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * On-device Boden-Segmentierung mit OpenCV.
 * User tippt auf den Boden → FloodFill segmentiert zusammenhängende Fläche gleicher Farbe.
 */
object LocalSegmenter {

    private var initialized = false

    private fun ensureInit() {
        if (!initialized) {
            initialized = OpenCVLoader.initLocal()
        }
    }

    data class SegmentResult(
        val maskBitmap: Bitmap,
        val pixelArea: Int,
        /** Pixel-Fläche bezogen auf die skalierte Version (für px/m Umrechnung) */
        val scaledPixelArea: Int,
        val scaleFactor: Float
    )

    data class FullImageSegment(
        val maskBitmap: Bitmap,
        val pixelArea: Int,
        val bboxFill: Double
    )

    /**
     * Segmentiert den Boden ab dem Punkt, auf den der User getippt hat.
     * @param bitmap Eingabe-Foto (RGB)
     * @param seedX X-Koordinate des Taps im Originalbild
     * @param seedY Y-Koordinate des Taps im Originalbild
     * @param tolerance Farbtoleranz (0-100), höher = mehr wird erfasst
     * @param maxDim Maximale Dimension für Performance
     */
    suspend fun segmentFromPoint(
        bitmap: Bitmap,
        seedX: Int,
        seedY: Int,
        tolerance: Int = 48,
        maxDim: Int = 512
    ): SegmentResult = withContext(Dispatchers.Default) {
        ensureInit()

        // 1) Bild skalieren
        val scale = maxDim.toFloat() / maxOf(bitmap.width, bitmap.height).toFloat()
        val w = (bitmap.width * scale).toInt().coerceAtLeast(2)
        val h = (bitmap.height * scale).toInt().coerceAtLeast(2)
        val scaled = Bitmap.createScaledBitmap(bitmap, w, h, true)

        // Seed-Point skalieren & clampen
        val sx = (seedX * scale).toInt().coerceIn(0, w - 1)
        val sy = (seedY * scale).toInt().coerceIn(0, h - 1)

        // 2) Bitmap → Mat (BGR)
        val src = Mat()
        Utils.bitmapToMat(scaled, src)
        Imgproc.cvtColor(src, src, Imgproc.COLOR_RGBA2BGR)

        // Leicht blurren um Rauschen zu reduzieren
        Imgproc.GaussianBlur(src, src, Size(5.0, 5.0), 0.0)

        // 3) HSV + CLAHE auf V-Kanal: lokale Helligkeitsangleichung gegen Schatten.
        //    FloodFill kann so über Schattenkanten hinweg dieselbe Farbflaeche
        //    erfassen, statt an der Beleuchtungs-Grenze zu stoppen.
        val hsv = Mat()
        Imgproc.cvtColor(src, hsv, Imgproc.COLOR_BGR2HSV)
        val hsvChannels = ArrayList<Mat>(3)
        Core.split(hsv, hsvChannels)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(hsvChannels[2], hsvChannels[2])
        Core.merge(hsvChannels, hsv)
        hsvChannels.forEach { it.release() }

        // 4) FloodFill von Seed-Point – füllt zusammenhängende ähnliche Farbe.
        //    Mask muss 2px groesser sein als Bild (OpenCV-Anforderung).
        val ffMask = Mat.zeros(Size((w + 2).toDouble(), (h + 2).toDouble()), CvType.CV_8UC1)

        val tH = (tolerance * 0.5).coerceAtLeast(8.0)
        val tS = tolerance.toDouble()
        val tV = (tolerance * 1.5).coerceAtLeast(35.0)  // V groszuegiger, CLAHE nimmt den Hauptteil
        val loDiff = Scalar(tH, tS, tV)
        val upDiff = Scalar(tH, tS, tV)

        Imgproc.floodFill(
            hsv, ffMask,
            Point(sx.toDouble(), sy.toDouble()),
            Scalar(0.0),
            Rect(),
            loDiff, upDiff,
            Imgproc.FLOODFILL_MASK_ONLY or Imgproc.FLOODFILL_FIXED_RANGE or (255 shl 8)
        )

        val floodMask = ffMask.submat(1, h + 1, 1, w + 1).clone()
        ffMask.release()

        // 5) Morphologie: Loecher schliessen, Streupixel raus
        val kernelClose = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(11.0, 11.0))
        val kernelOpen = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(floodMask, floodMask, Imgproc.MORPH_CLOSE, kernelClose)
        Imgproc.morphologyEx(floodMask, floodMask, Imgproc.MORPH_OPEN, kernelOpen)

        // 6) GrabCut-Refinement: snappt die Maske an die echten Bildkanten.
        //    FloodFill liefert "ungefaehr richtig" (farbbasiert), GrabCut zieht den
        //    Rand auf den tatsaechlichen Kontrastsprung. Bei Mis-Fit faellt der Code
        //    auf die FloodFill-Maske zurueck.
        val result = runGrabCutRefinement(src, floodMask, sx, sy, w, h) ?: floodMask.clone()
        floodMask.release()

        // Final cleanup auf der refined Maske
        Imgproc.morphologyEx(result, result, Imgproc.MORPH_CLOSE, kernelClose)

        // 7) Konturen finden & nur die groeszte behalten
        val contours = mutableListOf<MatOfPoint>()
        val contourSrc = result.clone()
        Imgproc.findContours(contourSrc, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        contourSrc.release()
        if (contours.isNotEmpty()) {
            val largest = contours.maxByOrNull { Imgproc.contourArea(it) }
            result.setTo(Scalar(0.0))
            if (largest != null) {
                // ── Form-Optimierung: Polygon-Approximation ──────────────────
                // Macht aus der FloodFill-Wolke eine geometrische Form.
                // 0.015-0.02 * Umfang ist ein guter Wert fuer "fast viereckig".
                val peri = Imgproc.arcLength(MatOfPoint2f(*largest.toArray()), true)
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(MatOfPoint2f(*largest.toArray()), approx, 0.015 * peri, true)
                
                val approxFixed = MatOfPoint(*approx.toArray())
                Imgproc.drawContours(result, listOf(approxFixed), -1, Scalar(255.0), -1)
                approxFixed.release(); approx.release()
            }
        }

        val scaledPixelArea = Core.countNonZero(result)
        val maskBitmap = createOverlayBitmap(result, bitmap.width, bitmap.height)
        val originalPixelArea = (scaledPixelArea / (scale * scale)).toInt()

        src.release(); hsv.release(); result.release()
        kernelClose.release(); kernelOpen.release()
        contours.forEach { it.release() }

        SegmentResult(maskBitmap, originalPixelArea, scaledPixelArea, scale)
    }

    /**
     * GrabCut-Refinement: nimmt eine grobe (farbbasierte) Maske und zieht den Rand
     * auf die echten Bildkanten. Definite-FGD = stark erodierte Eingangsmaske +
     * Seed-Punkt, Definite-BGD = weit auserhalb dilatierter Eingangsmaske,
     * dazwischen probable-FGD/BGD. Bei Fehlern (degenerierte Maske, OpenCV-
     * Exception) gibt's null zurueck, damit der Caller auf die FloodFill-Maske
     * zurueckfaellt.
     *
     * @param bgr 3-Kanal BGR-Bild (gleiche Groesze wie inputMask)
     * @param inputMask CV_8UC1, 0 oder 255
     */
    private fun runGrabCutRefinement(
        bgr: Mat, inputMask: Mat, seedX: Int, seedY: Int, w: Int, h: Int
    ): Mat? {
        val gcMask = Mat(h, w, CvType.CV_8UC1, Scalar(Imgproc.GC_PR_BGD.toDouble()))
        val sureFg = Mat()
        val seedFgd = Mat.zeros(h, w, CvType.CV_8UC1)
        val dilated = Mat()
        val sureBg = Mat()
        val kErode = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(9.0, 9.0))
        val kDilate = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(31.0, 31.0))
        val bgdModel = Mat()
        val fgdModel = Mat()

        return try {
            // Probable foreground = ganze FloodFill-Maske
            gcMask.setTo(Scalar(Imgproc.GC_PR_FGD.toDouble()), inputMask)

            // Definite foreground = stark erodiert + ein Punkt am Seed (Safety)
            Imgproc.erode(inputMask, sureFg, kErode)
            gcMask.setTo(Scalar(Imgproc.GC_FGD.toDouble()), sureFg)
            Imgproc.circle(seedFgd, Point(seedX.toDouble(), seedY.toDouble()), 3, Scalar(255.0), -1)
            gcMask.setTo(Scalar(Imgproc.GC_FGD.toDouble()), seedFgd)

            // Definite background = weit auserhalb der dilatierten FloodFill-Maske
            Imgproc.dilate(inputMask, dilated, kDilate)
            Core.bitwise_not(dilated, sureBg)
            gcMask.setTo(Scalar(Imgproc.GC_BGD.toDouble()), sureBg)

            Imgproc.grabCut(bgr, gcMask, Rect(), bgdModel, fgdModel, 2, Imgproc.GC_INIT_WITH_MASK)

            // FGD || PR_FGD → 255, sonst 0
            val out = Mat()
            val tmpFgd = Mat()
            val tmpPrFgd = Mat()
            Core.compare(gcMask, Scalar(Imgproc.GC_FGD.toDouble()), tmpFgd, Core.CMP_EQ)
            Core.compare(gcMask, Scalar(Imgproc.GC_PR_FGD.toDouble()), tmpPrFgd, Core.CMP_EQ)
            Core.bitwise_or(tmpFgd, tmpPrFgd, out)
            tmpFgd.release(); tmpPrFgd.release()
            out
        } catch (t: Throwable) {
            android.util.Log.w("LocalSegmenter", "GrabCut failed, fallback to FloodFill mask: ${t.message}")
            null
        } finally {
            gcMask.release(); sureFg.release(); seedFgd.release()
            dilated.release(); sureBg.release()
            kErode.release(); kDilate.release()
            bgdModel.release(); fgdModel.release()
        }
    }

    /**
     * Stage-2-Variante: FloodFill innerhalb einer vorhandenen Maske (Stage 1).
     * Alle Pixel auserhalb stage1Mask werden vorher geschwaerzt, das Ergebnis
     * danach mit der gleichen Stage-1-Maske geschnitten — so kann der FloodFill
     * niemals auf den Hintergrund auserhalb der Platte uberspringen.
     *
     * Arbeitet komplett auf Skalen-Aufloesung (maxDim=512 default), nicht auf
     * Foto-Originalgroesse. Sonst frisst der Aufruf bei 12-MP-Fotos mehrere
     * hundert MB Heap (5-6 parallele Int-Arrays) und der Prozess wird vom OS
     * gekillt.
     */
    suspend fun segmentFromPointWithinMask(
        bitmap: Bitmap,
        stage1Mask: Bitmap,
        seedX: Int,
        seedY: Int,
        tolerance: Int = 48,
        maxDim: Int = 512
    ): SegmentResult = withContext(Dispatchers.Default) {
        val origW = bitmap.width
        val origH = bitmap.height
        val scale = maxDim.toFloat() / maxOf(origW, origH).toFloat()
        // Wenn das Bild eh kleiner ist als maxDim: scale = 1 und wir arbeiten direkt.
        val sw = (origW * scale.coerceAtMost(1f)).toInt().coerceAtLeast(2)
        val sh = (origH * scale.coerceAtMost(1f)).toInt().coerceAtLeast(2)

        // 1) Bitmap + Stage1-Maske auf Skalengroesse runter
        val scaledBmp = if (sw == origW && sh == origH) bitmap
                         else Bitmap.createScaledBitmap(bitmap, sw, sh, true)
        val scaledMask = if (stage1Mask.width == sw && stage1Mask.height == sh) stage1Mask
                          else Bitmap.createScaledBitmap(stage1Mask, sw, sh, false)

        // 2) Maskiertes Foto: Pixel auserhalb Stage1 → Schwarz (alles bei Skalengroesse)
        val srcPx = IntArray(sw * sh); scaledBmp.getPixels(srcPx, 0, sw, 0, 0, sw, sh)
        val mskPx = IntArray(sw * sh); scaledMask.getPixels(mskPx, 0, sw, 0, 0, sw, sh)
        val maskedPx = IntArray(sw * sh)
        for (i in srcPx.indices) {
            val inside = ((mskPx[i] ushr 24) and 0xFF) > 16
            maskedPx[i] = if (inside) srcPx[i] else 0xFF000000.toInt()
        }
        if (scaledBmp !== bitmap) scaledBmp.recycle()

        val maskedBmp = Bitmap.createBitmap(sw, sh, Bitmap.Config.ARGB_8888)
        maskedBmp.setPixels(maskedPx, 0, sw, 0, 0, sw, sh)

        // 3) Seed in Skalen-Koords
        val ssx = (seedX * (sw.toFloat() / origW)).toInt().coerceIn(0, sw - 1)
        val ssy = (seedY * (sh.toFloat() / origH)).toInt().coerceIn(0, sh - 1)

        // 4) FloodFill auf maskiertem Bild (kein weiteres Scaling noetig)
        val seg = segmentFromPoint(maskedBmp, ssx, ssy, tolerance, maxDim)
        maskedBmp.recycle()

        // 5) Ergebnis-Maske mit Stage1 schneiden — beide jetzt in Skalengroesse
        val resultMask = seg.maskBitmap
        val rw = resultMask.width; val rh = resultMask.height
        val rPx = IntArray(rw * rh); resultMask.getPixels(rPx, 0, rw, 0, 0, rw, rh)

        val gateBmp = if (scaledMask.width == rw && scaledMask.height == rh) scaledMask
                       else Bitmap.createScaledBitmap(scaledMask, rw, rh, false)
        val gPx = IntArray(rw * rh); gateBmp.getPixels(gPx, 0, rw, 0, 0, rw, rh)

        val finalPx = IntArray(rw * rh)
        var count = 0
        for (i in rPx.indices) {
            val t = ((rPx[i] ushr 24) and 0xFF) > 16
            val g = ((gPx[i] ushr 24) and 0xFF) > 16
            if (t && g) { finalPx[i] = rPx[i]; count++ } else finalPx[i] = 0
        }
        if (gateBmp !== scaledMask) gateBmp.recycle()
        if (scaledMask !== stage1Mask) scaledMask.recycle()
        resultMask.recycle()

        val finalBmp = Bitmap.createBitmap(rw, rh, Bitmap.Config.ARGB_8888)
        finalBmp.setPixels(finalPx, 0, rw, 0, 0, rw, rh)
        // pixelArea ist in Skalen-Pixeln. Wenn der Caller in Original-Pixeln rechnen
        // moechte: count / (scale * scale).
        SegmentResult(finalBmp, count, count, scale)
    }

    /**
     * Zerlegt das gesamte Bild in mehrere zusammenhängende, auswählbare Regionen.
     * Anders als der Punkt-Scan deckt dieser Pfad das komplette Foto ab.
     */
    suspend fun segmentWholeImage(
        bitmap: Bitmap,
        maxDim: Int = 480,
        minAreaNorm: Double = 0.0018,
        maxAreaNorm: Double = 0.82,
        maxRegions: Int = 24
    ): List<FullImageSegment> = withContext(Dispatchers.Default) {
        ensureInit()
        if (bitmap.width < 8 || bitmap.height < 8) return@withContext emptyList()

        val scale = (maxDim.toFloat() / maxOf(bitmap.width, bitmap.height).toFloat()).coerceAtMost(1f)
        val sw = (bitmap.width * scale).toInt().coerceAtLeast(8)
        val sh = (bitmap.height * scale).toInt().coerceAtLeast(8)
        val scaled = if (sw == bitmap.width && sh == bitmap.height) bitmap
                     else Bitmap.createScaledBitmap(bitmap, sw, sh, true)

        val src = Mat()
        Utils.bitmapToMat(scaled, src)
        Imgproc.cvtColor(src, src, Imgproc.COLOR_RGBA2RGB)
        val shifted = Mat()
        Imgproc.pyrMeanShiftFiltering(src, shifted, 14.0, 20.0, 1)
        Imgproc.cvtColor(shifted, shifted, Imgproc.COLOR_RGB2RGBA)

        val smoothed = Bitmap.createBitmap(sw, sh, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(shifted, smoothed)
        val px = IntArray(sw * sh)
        smoothed.getPixels(px, 0, sw, 0, 0, sw, sh)

        val visited = BooleanArray(sw * sh)
        val labels = IntArray(sw * sh) { -1 }
        val queue = IntArray(sw * sh)
        data class Candidate(val id: Int, val area: Int, val minX: Int, val minY: Int, val maxX: Int, val maxY: Int, val fill: Double)
        val candidates = mutableListOf<Candidate>()
        var nextId = 0

        fun colorDist(a: Int, b: Int): Int {
            val ar = (a shr 16) and 0xFF; val ag = (a shr 8) and 0xFF; val ab = a and 0xFF
            val br = (b shr 16) and 0xFF; val bg = (b shr 8) and 0xFF; val bb = b and 0xFF
            return kotlin.math.abs(ar - br) + kotlin.math.abs(ag - bg) + kotlin.math.abs(ab - bb)
        }

        val minPixels = (sw * sh * minAreaNorm).toInt().coerceAtLeast(24)
        val maxPixels = (sw * sh * maxAreaNorm).toInt().coerceAtLeast(minPixels + 1)
        for (start in px.indices) {
            if (visited[start]) continue
            visited[start] = true
            val seedColor = px[start]
            var head = 0
            var tail = 0
            queue[tail++] = start
            var area = 0
            var minX = sw
            var minY = sh
            var maxX = 0
            var maxY = 0

            while (head < tail) {
                val idx = queue[head++]
                labels[idx] = nextId
                area++
                val x = idx % sw
                val y = idx / sw
                if (x < minX) minX = x
                if (y < minY) minY = y
                if (x > maxX) maxX = x
                if (y > maxY) maxY = y

                if (x > 0) {
                    val n = idx - 1
                    if (!visited[n] && colorDist(seedColor, px[n]) <= 44) {
                        visited[n] = true
                        queue[tail++] = n
                    }
                }
                if (x < sw - 1) {
                    val n = idx + 1
                    if (!visited[n] && colorDist(seedColor, px[n]) <= 44) {
                        visited[n] = true
                        queue[tail++] = n
                    }
                }
                if (y > 0) {
                    val n = idx - sw
                    if (!visited[n] && colorDist(seedColor, px[n]) <= 44) {
                        visited[n] = true
                        queue[tail++] = n
                    }
                }
                if (y < sh - 1) {
                    val n = idx + sw
                    if (!visited[n] && colorDist(seedColor, px[n]) <= 44) {
                        visited[n] = true
                        queue[tail++] = n
                    }
                }
            }

            val bboxArea = ((maxX - minX + 1) * (maxY - minY + 1)).coerceAtLeast(1)
            val fill = area.toDouble() / bboxArea.toDouble()
            if (area in minPixels..maxPixels && fill >= 0.14) {
                candidates += Candidate(nextId, area, minX, minY, maxX, maxY, fill)
            }
            nextId++
        }

        val selected = candidates
            .sortedWith(compareByDescending<Candidate> { it.area }.thenByDescending { it.fill })
            .take(maxRegions)

        val result = selected.map { candidate ->
            val maskPixels = IntArray(sw * sh)
            for (i in labels.indices) {
                if (labels[i] == candidate.id) {
                    maskPixels[i] = Color.argb(255, 255, 255, 255)
                }
            }
            val maskSmall = Bitmap.createBitmap(sw, sh, Bitmap.Config.ARGB_8888)
            maskSmall.setPixels(maskPixels, 0, sw, 0, 0, sw, sh)
            val maskFull = if (sw == bitmap.width && sh == bitmap.height) maskSmall
                           else Bitmap.createScaledBitmap(maskSmall, bitmap.width, bitmap.height, false).also { maskSmall.recycle() }
            FullImageSegment(maskFull, candidate.area, candidate.fill)
        }

        src.release()
        shifted.release()
        smoothed.recycle()
        if (scaled !== bitmap) scaled.recycle()
        result
    }

    /**
     * Berechnet den Pixel-Abstand zwischen zwei Punkten im Bild.
     * Für Referenz-Kalibrierung: User tippt zwei Punkte auf einem Zollstock.
     */
    fun pixelDistance(x1: Float, y1: Float, x2: Float, y2: Float): Double {
        val dx = (x2 - x1).toDouble()
        val dy = (y2 - y1).toDouble()
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    /**
     * Erstellt eine transparente Overlay-Bitmap aus der binären Maske.
     */
    private fun createOverlayBitmap(mask: Mat, targetW: Int, targetH: Int): Bitmap {
        val resized = Mat()
        Imgproc.resize(mask, resized, Size(targetW.toDouble(), targetH.toDouble()))

        val overlay = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(targetW * targetH)
        val maskData = ByteArray(targetW * targetH)
        resized.get(0, 0, maskData)

        for (i in pixels.indices) {
            pixels[i] = if ((maskData[i].toInt() and 0xFF) > 0) {
                Color.argb(120, 0, 200, 80)
            } else {
                Color.TRANSPARENT
            }
        }
        overlay.setPixels(pixels, 0, targetW, 0, 0, targetW, targetH)
        resized.release()
        return overlay
    }
}