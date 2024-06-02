//
//  MainViewController.swift
//  healthSignal
//
//  Created by Emir Can Tanyıldız on 26.05.2024.
//

import UIKit

class MainViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        
        imageView.image = UIImage(named: "weekday-three-doctors-walking-forward")
    }
    
    @IBAction func nextButton(_ sender: Any) {
        performSegue(withIdentifier: "toBrainVC", sender: nil)
    }
    
    
    @IBOutlet weak var imageView: UIImageView!
    


}
