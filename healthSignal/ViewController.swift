import UIKit
import WebKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, WKNavigationDelegate, WKScriptMessageHandler {
    @IBOutlet weak var webView: WKWebView!
    @IBOutlet weak var originalImage: UIImageView!
    
    @IBOutlet weak var resultLabel: UILabel!
    
    let imagePicker = UIImagePickerController()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Web view configuration
        let contentController = WKUserContentController()
        contentController.add(self, name: "resultHandler")
        
        let config = WKWebViewConfiguration()
        config.userContentController = contentController
        
        webView = WKWebView(frame: self.view.bounds, configuration: config)
        webView.navigationDelegate = self
        self.view.addSubview(webView)
        
        webView.load(URLRequest(url: URL(string: "http://127.0.0.1:5000/")!))
        
        // Image picker configuration
        imagePicker.delegate = self
    }
    
    @IBAction func selectImage(_ sender: UIButton) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        picker.allowsEditing = true
        present(picker, animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let editedImage = info[.editedImage] as? UIImage {
            uploadImage(image: editedImage)
        } else if let originalImage = info[.originalImage] as? UIImage {
            uploadImage(image: originalImage)
        }
        dismiss(animated: true, completion: nil)
    }
    
    func uploadImage(image: UIImage) {
        guard let imageData = image.jpegData(compressionQuality: 1.0) else { return }
        let base64String = imageData.base64EncodedString()
        
        let script = """
        var input = document.querySelector('input[type="file"]');
        var file = new File(['\(base64String)'], 'image.jpg', { type: 'image/jpeg' });
        var dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;
        var button = document.querySelector('button');
        button.click();
        """
        
        webView.evaluateJavaScript(script) { (result, error) in
            if let error = error {
                print("Error injecting script: \(error)")
            } else {
                self.extractResult()
            }
        }
    }
    
    func extractResult() {
        let script = """
        setTimeout(function() {
            var resultText = document.querySelector('h3').innerText;
            window.webkit.messageHandlers.resultHandler.postMessage(resultText);
        }, 5000);
        """
        
        webView.evaluateJavaScript(script, completionHandler: nil)
    }
    
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        if message.name == "resultHandler", let resultText = message.body as? String {
            resultLabel.text = resultText
        }
    }
}





