package com.example.edgerton_air_soft.segmentation

import android.graphics.Bitmap

/**
 * Ein einzelnes Sub-Ziel auf der bereits segmentierten Zielplatte (Stage 2).
 * Wird vom Multi-Tap-Workflow erzeugt: User tippt das Ziel an, EfficientSAM
 * liefert eine saubere Maske innerhalb der Stage-1-Region.
 *
 * @param id            Stabile ID zum Identifizieren in Listen / UI (1-basiert)
 * @param centerXNorm   Tap-Position X in [0..1] der Originalbild-Aufloesung
 * @param centerYNorm   Tap-Position Y in [0..1] der Originalbild-Aufloesung
 * @param maskBitmap    ARGB-Maske (Alpha > 16 = Foreground) in Originalgroesse,
 *                      oder null falls die Segmentierung noch nicht erfolgt oder
 *                      fehlgeschlagen ist (dann zaehlt nur centerX/Y als Hint).
 * @param areaNorm      Maskenflaeche normalisiert auf das ganze Bild
 *                      (Foreground-Pixel / GesamtPixel). 0.0 wenn maskBitmap null.
 * @param label         Optionaler User-Name (z.B. "links oben", "Bullseye"); leer = nur ID anzeigen.
 */
data class TargetEntry(
    val id: Int,
    val centerXNorm: Double,
    val centerYNorm: Double,
    val maskBitmap: Bitmap? = null,
    val areaNorm: Double = 0.0,
    val label: String = ""
)
