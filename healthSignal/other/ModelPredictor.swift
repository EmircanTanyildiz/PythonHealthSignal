import UIKit
import CoreML
import Vision

class ModelPredictor {
    
    private lazy var model: BrainTumorModel? = {
        guard let model = try? BrainTumorModel(configuration: MLModelConfiguration()) else {
            print("BrainTumorModel yüklenemedi.")
            return nil
        }
        return model
    }()
    
    func getResult(image: UIImage, completion: @escaping (Int?) -> Void) {
        guard let model = self.model,
              let resizedImage = resize(image, to: CGSize(width: 224, height: 224)),
              let pixelBuffer = preprocessImage(resizedImage) else {
            print("Görüntü ön işleme başarısız oldu.")
            completion(nil)
            return
        }
        
        do {
            let output = try model.prediction(input_1: pixelBuffer)  // Modelinizin giriş adı doğru mu kontrol edin.
            guard let featureValue = output.featureValue(for: "output1"),  // Modelinizin çıkış adı doğru mu kontrol edin.
                  let multiArrayValue = featureValue.multiArrayValue else {
                print("Beklenen çıktı bulunamadı.")
                completion(nil)
                return
            }

            let pred = multiArrayValue[0].doubleValue
            let result = pred > 0.5 ? 0 : 1
            completion(result)
        } catch {
            print("Tahmin yapılırken bir hata oluştu: \(error.localizedDescription)")
            completion(nil)
        }
    }
    
    private func resize(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        defer { UIGraphicsEndImageContext() }
        image.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    private func preprocessImage(_ image: UIImage) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let options: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferWidthKey: 224,
            kCVPixelBufferHeightKey: 224
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, 224, 224, kCVPixelFormatType_32ARGB, options as CFDictionary, &pixelBuffer)
        
        guard let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData, width: 224, height: 224, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            return nil
        }
        
        context.translateBy(x: 0, y: 224)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        image.draw(in: CGRect(x: 0, y: 0, width: 224, height: 224))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}

extension UIImage {
    func getPixelColor(pos: CGPoint) -> (red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8)? {
        guard let cgImage = self.cgImage else { return nil }
        let pixelData = cgImage.dataProvider?.data
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        
        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y)) + Int(pos.x)) * 4
        
        let r = data[pixelInfo]
        let g = data[pixelInfo+1]
        let b = data[pixelInfo+2]
        let a = data[pixelInfo+3]
        
        return (r, g, b, a)
    }
}
extension BrainTumorModel {
    func prediction(inputImage: CVPixelBuffer) throws -> BrainTumorModelOutput {
        let model = try BrainTumorModel(configuration: MLModelConfiguration())
        return try model.prediction(inputImage: inputImage)
    }
}
