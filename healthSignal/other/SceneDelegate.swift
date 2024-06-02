//
//  SceneDelegate.swift
//  healthSignal
//
//  Created by Emir Can Tanyıldız on 22.05.2024.
//
import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {

    var window: UIWindow?

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = (scene as? UIWindowScene) else { return }
        let window = UIWindow(windowScene: windowScene)
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        // LoginViewController'ın kimliği ile view controller'ı yükleyin
        let loginViewController = storyboard.instantiateViewController(withIdentifier: "LoginViewController") as! LoginViewController
        window.rootViewController = loginViewController
        self.window = window
        window.makeKeyAndVisible()
    }
}
