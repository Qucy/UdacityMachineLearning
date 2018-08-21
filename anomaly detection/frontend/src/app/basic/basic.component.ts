import { Router, ActivatedRoute } from '@angular/router';
import { Component, OnInit,TemplateRef } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from "@angular/forms";
import { HackathonService } from "../services/hackathon.service";
import { Ng5SliderModule } from 'ng5-slider';
import { Hackathon } from "../model/hackathon.model";
import { HackathonTest } from "../model/test.model";
import { BsModalService } from 'ngx-bootstrap/modal';
import { BsModalRef } from 'ngx-bootstrap/modal/bs-modal-ref.service'; 


@Component({
  selector: 'app-root',
  templateUrl: './basic.component.html',
  styleUrls: ['./basic.component.scss']
})
export class BasicComponent implements OnInit{
  title = 'hackathonPortal';
  constructor(private formBuilder: FormBuilder, 
    private hackathonService: HackathonService,
    private router: Router,
    private modalService: BsModalService) { }
  form: FormGroup;
  hackathon: Hackathon = new Hackathon(); 
  start:number = 1;
  testArr: Array<HackathonTest> = []
  predictResult:string ;
  options = {
    floor: 0,
    ceil: 100
  };
  timeOptions = {
    floor: 1,
    ceil: 24
  }; 
  bsModalRef: BsModalRef;
  ngOnInit() { 
    this.form = this.formBuilder.group({ 
      time: ['1'],
      cpu: ['1', Validators.required],
      network: ['1', Validators.required]
    });

    let test1 = new HackathonTest();
    test1.time = "03:00";
    test1.cpuRange = "10~20";
    test1.networkRange = "10~20";
    test1.cpu = 15;
    test1.network = 15;
    test1.expect = 1;
    let test2 = new HackathonTest();
    test2.time = "03:00";
    test2.cpuRange = "10~20";
    test2.networkRange = "10~20";
    test2.cpu = 32;
    test2.network = 32;
    test2.expect = -1;
    let test3 = new HackathonTest();
    test3.time = "08:00";
    test3.cpuRange = "20~30";
    test3.networkRange = "20~30";
    test3.cpu = 25;
    test3.network = 25;
    test3.expect = 1;
    let test4 = new HackathonTest();
    test4.time = "08:00";
    test4.cpuRange = "20~30";
    test4.networkRange = "20~30";
    test4.cpu = 40;
    test4.network = 40;
    test4.expect = -1;
    let test5 = new HackathonTest();
    test5.time = "12:00";
    test5.cpuRange = "30~40";
    test5.networkRange = "30~40";
    test5.cpu = 35;
    test5.network = 35;
    test5.expect = 1;
    let test6 = new HackathonTest();
    test6.time = "12:00";
    test6.cpuRange = "30~40";
    test6.networkRange = "30~40";
    test6.cpu = 50;
    test6.network = 50;
    test6.expect = -1;
    let test7 = new HackathonTest();
    test7.time = "20:00";
    test7.cpuRange = "20~30";
    test7.networkRange = "20~30";
    test7.cpu = 25;
    test7.network = 25;
    test7.expect = 1;
    let test8 = new HackathonTest();
    test8.time = "20:00";
    test8.cpuRange = "20~30";
    test8.networkRange = "20~30";
    test8.cpu = 40;
    test8.network = 40;
    test8.expect = -1;
    let test9 = new HackathonTest();
    test9.time = "23:00";
    test9.cpuRange = "10~20";
    test9.networkRange = "10~20";
    test9.cpu = 15;
    test9.network = 15;
    test9.expect = 1;
    let test10 = new HackathonTest();
    test10.time = "23:00";
    test10.cpuRange = "10~20";
    test10.networkRange = "10~20";
    test10.cpu = 35;
    test10.network = 35;
    test10.expect = -1; 
    // this.testArr.push(test1);
    // this.testArr.push(test2);    
    this.testArr.push(test3);
    this.testArr.push(test4);
    this.testArr.push(test5);
    this.testArr.push(test6);
    this.testArr.push(test7);
    this.testArr.push(test8);
    // this.testArr.push(test9);
    // this.testArr.push(test10);   
   
  }

  predict() {
    this.hackathonService.setParam(this.hackathon)
      .subscribe(data => {
        if(parseInt(data)==1){
          this.predictResult = "Normal";
        }else{
          this.predictResult = "Anomaly";
        } 
      });
  }

  train() {
    // if(this.start == 1){
    //   this.changepic();
    // }
    this.hackathonService.basicTrain()
    .subscribe((rs) => { 
      alert("training complete!");
      if(this.start == 1){
        this.changepic();
      }
      this.start ++ ;
    },
    (error) => {
      if(error.status == 200){
        console.log("basic taining success!")
      } 
    });
    
  }

  t = new Array("http://100.98.136.15:5000/static/plot1.jpg", "http://100.98.136.15:5000/static/plot2.jpg", "http://100.98.136.15:5000/static/plot3.jpg", "http://100.98.136.15:5000/static/plot4.jpg",
  "http://100.98.136.15:5000/static/plot5.jpg", "http://100.98.136.15:5000/static/plot6.jpg", "http://100.98.136.15:5000/static/plot7.jpg", "http://100.98.136.15:5000/static/plot8.jpg",
  "http://100.98.136.15:5000/static/plot9.jpg", "http://100.98.136.15:5000/static/plot10.jpg", "http://100.98.136.15:5000/static/plot11.jpg", "http://100.98.136.15:5000/static/plot12.jpg",
  "http://100.98.136.15:5000/static/plot13.jpg", "http://100.98.136.15:5000/static/plot14.jpg", "http://100.98.136.15:5000/static/plot15.jpg", "http://100.98.136.15:5000/static/plot16.jpg",);
  i = 0;
  imgUrl = "../assets/picture/hackathon.jpg";
  testResultImgUrl = "";
  flag:number = 1;
  //循环方法
  changepic() { 
    this.imgUrl = this.t[this.i]+"?"+new Date().getTime(); 
    if(this.flag==1 && this.i==15){
      this.flag = 0;
    }else if(this.flag==1 && this.i<15){
      this.flag = 1;
    }else if(this.flag==0 && this.i>0){
      this.flag = 0;
    }else if(this.flag==0 && this.i==0){
      this.flag = 1;
    }
    
    if(this.flag==0){
      this.i--; 
    }else{
      this.i++;
    }
    setTimeout(() => {
      this.changepic()
    }, 500);
  }

  goToAuto() {
    this.router.navigate(['/auto']);
  }

  benchmark(){ 
    this.hackathonService.benchmark()
    .subscribe(data => {    
      let predList:Array<number> = data.split(","); 
      let i = 0;
      for(let model of this.testArr){
        model.predict = predList[i]; 
        i++;  
      }
    });
  }

  openModal(template: TemplateRef<any>) {
    this.testResultImgUrl = "http://100.98.136.15:5000/static/plot0.jpg?"+new Date().getTime();
    this.bsModalRef = this.modalService.show(template);
  }
}


