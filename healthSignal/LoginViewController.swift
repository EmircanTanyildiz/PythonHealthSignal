//
//  LoginViewController.swift
//  healthSignal
//
//  Created by Emir Can Tanyıldız on 26.05.2024.
//

import UIKit

class LoginViewController: UIViewController {
    var myUserName = [String]()
    var myPassword = [String]()
    @IBOutlet weak var usernameTextField: UITextField!
    @IBOutlet weak var passTestField: UITextField!
    @IBOutlet weak var tekrarSifreTextField: UITextField!

    func error(mainTit: String,tit : String, okMes : String) {
        let uyariMesaji =  UIAlertController(title: mainTit, message: tit, preferredStyle: UIAlertController.Style.alert)
        let okButton = UIAlertAction(title: okMes, style: UIAlertAction.Style.default) { UIAlertAction in
        }
        uyariMesaji.addAction(okButton)
        self.present(uyariMesaji, animated: true, completion: nil)
    }
    var countOfEmp: Int = 1
    override func viewDidLoad() {
        super.viewDidLoad()
        tekrarSifreTextField.isHidden = true
        myUserName.append("emircantanyildiz")
        myPassword.append("123456")
    }
    var loginUSER = [String]()
    @IBAction func loginButton(_ sender: Any) {
        var x = 0
        var dogruluk = false
        while x < countOfEmp {
            if usernameTextField.text == myUserName[x] && passTestField.text == myPassword[x]{
                dogruluk = true
                loginUSER.append(myUserName[x])
                x = countOfEmp
            }
            x += 1
        }
        if dogruluk == false{
            error(mainTit: "Health Signal'dan Mesaj Var", tit: "Giris Yapılamadı", okMes: "Okey 👍")
        }
        performSegue(withIdentifier: "toPage1VC", sender:  nil)
    }
    
    @IBAction func registerButton(_ sender: Any) {
            tekrarSifreTextField.isHidden = false
            
            if tekrarSifreTextField.text == "" || usernameTextField.text == "" || passTestField.text == "" {
                error(mainTit: "Health Signal'den Mesaj", tit: "Kullanıcı isminizi ve Parolanızı girin", okMes: "Tamamdır 👍")
            }
           
            print(usernameTextField.text)
            print(passTestField.text)
            print(tekrarSifreTextField.text)
            if passTestField.text != tekrarSifreTextField.text || tekrarSifreTextField.text == ""{
                error(mainTit: "Health Signal'den Mesaj", tit: "Parolarınız Uyuşmuyor", okMes: "Düzeltiyorum 👍")
            }else{
                myUserName.append(usernameTextField.text!)
                myPassword.append(passTestField.text!)
                tekrarSifreTextField.isHidden = true
                usernameTextField.text = ""
                passTestField.text = ""
                countOfEmp += 1
            }
        
       
        
        
        
    }
    

}
