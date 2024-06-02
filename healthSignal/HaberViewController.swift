//
//  NewsViewController.swift
//  healthSignal
//
//  Created by Emir Can Tanyıldız on 26.05.2024.
//

import UIKit

class HaberViewController: UIViewController, UITableViewDelegate, UITableViewDataSource {
    @IBOutlet weak var tableVC5: UITableView!
    private var newsTableViewModel :NewsTableViewModel!
    override func viewDidLoad() {
        super.viewDidLoad()
        tableVC5.delegate = self
        tableVC5.dataSource = self
        tableVC5.rowHeight = UITableView.automaticDimension
        tableVC5.estimatedRowHeight = UITableView.automaticDimension
        veriAl()
    }
    func veriAl(){
        let url = URL(string: "https://raw.githubusercontent.com/EmircanTanyildiz/Cambio/main/news.json")
        webService().haberleriIndir(url: url!) { (haberler) in
            if let haberler = haberler{
                self.newsTableViewModel = NewsTableViewModel(newList: haberler)
                DispatchQueue.main.async {
                    self.tableVC5.reloadData()
                }
            }
        }
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return newsTableViewModel == nil ? 0 : self.newsTableViewModel.numberOfRowsInSection()
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "Cell", for: indexPath) as! newsCell
        let newsViewModel = self.newsTableViewModel.newsAtIndexPath(indexPath.row)
        cell.titleLabel.text = newsViewModel.title
        cell.storyLabel.text = newsViewModel.story
        return cell
    }
    

    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return UITableView.automaticDimension
    }

}
