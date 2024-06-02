import UIKit
class Page1ViewController: UIViewController {

    @IBOutlet weak var welcomeLabel: UILabel!
    @IBOutlet weak var imageView1: UIImageView!
    var countOfClick = 0
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView1.image = UIImage(named: "martina-mans-face-in-a-circle-1")
        welcomeLabel.text = "Hoşgeldiniz"
    }
    
    @IBAction func nextButton(_ sender: Any) {
        print("Next Button Tıklandı")
        switch countOfClick {
        case 0:
            imageView1.image = UIImage(named: "lounge-doctor-with-syringe-about-to-vaccinate-a-patient")
            welcomeLabel.text = "Health Signal Emrinizde"
            countOfClick += 1
        case 1:
            imageView1.image = UIImage(named: "3d-business-man-raising-cup")
            welcomeLabel.text = "Yapay Zeka kullanarak %95.16 Doğruluk oranı ile Beyin Tümörü Tespiti yapabilirsin"
            countOfClick += 1
        case 2:
            imageView1.image = UIImage(named: "lounge-woman-taking-a-picture-of-herself-in-the-mirror")
            welcomeLabel.text = "Telefonunundan Fotoğraf çekerek Saniyeler içerinde Tümör tespit edebilirsin (Beta)"
            countOfClick += 1
            
        default:
            print("")
            //Buraya Perform Segue koymayı Unutma
            performSegue(withIdentifier: "toMainMenuVC", sender: nil)
        }
    }
    
}
