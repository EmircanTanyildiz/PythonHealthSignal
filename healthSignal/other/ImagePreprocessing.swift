//
//  ImagePreprocessing.swift
//  healthSignal
//
//  Created by Emir Can Tanyıldız on 24.05.2024.
import UIKit

class ImagePreprocessing1 {
    static func preprocessImage(image: UIImage, targetSize: CGSize) -> CVPixelBuffer? {
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 2.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        guard let cgImage = resizedImage?.cgImage else { return nil }

        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext()

        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        let width = Int(targetSize.width)
        let height = Int(targetSize.height)

        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard let unwrappedPixelBuffer = pixelBuffer else { return nil }

        context.render(ciImage, to: unwrappedPixelBuffer)

        return unwrappedPixelBuffer
    }
}
