package com.example.edgerton_air_soft.segmentation

import android.graphics.Bitmap
import android.graphics.Color
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Two-Stage Segmenter:
 * Performs segmentation on the original image, then performs a second segmentation
 * restricted only to the area found in the first pass.
 */
object TwoStageSegmenter {

    data class TwoStageResult(
        val stage1Overlay: Bitmap, // Green overlay (original)
        val stage2Overlay: Bitmap, // Red overlay (restricted)
        val stage1Area: Int,
        val stage2Area: Int
    )

    /**
     * Performs two stages of segmentation.
     * @param bitmap The original source image.
     * @param seedX1 X coordinate for the first tap.
        * @param seedY1 Y coordinate for the first tap.
     * @param seedX2 X coordinate for the second tap (within stage 1 area).
     * @param seedY2 Y coordinate for the second tap.
     * @param tolerance Tolerance for both stages.
     */
    suspend fun segmentTwoStages(
        bitmap: Bitmap,
        seedX1: Int,
        seedY1: Int,
        seedX2: Int,
        seedY2: Int,
        tolerance: Int = 35
    ): TwoStageResult? = withContext(Dispatchers.Default) {
        try {
            // --- STAGE 1: Standard Segmentation ---
            val stage1 = LocalSegmenter.segmentFromPoint(bitmap, seedX1, seedY1, tolerance)

            // Create a "masked" bitmap for Stage 2 (only pixels from Stage 1 are kept)
            val maskedBitmap = createMaskedBitmap(bitmap, stage1.maskBitmap)

            // --- STAGE 2: Restricted Segmentation ---
            val stage2 = LocalSegmenter.segmentFromPoint(maskedBitmap, seedX2, seedY2, tolerance)

            // Scale Stage 2 overlay back to original size and color it Red
            val stage2OverlayOnOriginal = scaleAndColorOverlay(stage2.maskBitmap, bitmap.width, bitmap.height, Color.argb(150, 255, 0, 0))

            TwoStageResult(
                stage1Overlay = stage1.maskBitmap,
                stage2Overlay = stage2OverlayOnOriginal,
                stage1Area = stage1.pixelArea,
                stage2Area = stage2.pixelArea
            )
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun createMaskedBitmap(original: Bitmap, maskOverlay: Bitmap): Bitmap {
        val width = original.width
        val height = original.height
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        val srcPixels = IntArray(width * height)
        val maskPixels = IntArray(width * height)

        original.getPixels(srcPixels, 0, width, 0, 0, width, height)
        maskOverlay.getPixels(maskPixels, 0, width, 0, 0, width, height)

        val outPixels = IntArray(width * height)

        for (i in srcPixels.indices) {
            // Check if the pixel is part of the mask (using green channel as indicator for hit)
            val isHit = (maskPixels[i] and 0x00FF0000.toInt()) > 0 || (maskPixels[i] and 0x0000FF00.toInt()) > 0

            if (isHit) {
                outPixels[i] = srcPixels[i]
            } else {
                outPixels[i] = Color.BLACK
            }
        }

        result.setPixels(outPixels, 0, width, 0, 0, width, height)
        return result
    }

    private fun scaleAndColorOverlay(mask: Bitmap, targetW: Int, targetH: Int, newColor: Int): Bitmap {
        val resized = Bitmap.createScaledBitmap(mask, targetW, targetH, true)
        val overlay = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(targetW * targetH)
        resized.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH)

        val outPixels = IntArray(targetW * targetH)
        for (i in pixels.indices) {
            if ((pixels[i] and 0xFF) > 0 || (pixels[i] and 0x00FF0000.toInt()) > 0) {
                outPixels[i] = newColor
            } else {
                outPixels[i] = Color.TRANSPARENT
            }
        }

        overlay.setPixels(outPixels, 0, targetW, 0, 0, targetW, targetH)
        resized.recycle()
        return overlay
    }
}