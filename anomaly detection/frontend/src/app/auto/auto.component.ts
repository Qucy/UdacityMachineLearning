import { Component, TemplateRef } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from "@angular/forms";
import { HackathonService } from "../services/hackathon.service";
import { Ng5SliderModule } from 'ng5-slider';
import { Hackathon } from "../model/hackathon.model";
import { Router, ActivatedRoute } from '@angular/router';
import { BsModalService } from 'ngx-bootstrap/modal';
import { BsModalRef } from 'ngx-bootstrap/modal/bs-modal-ref.service';

@Component({
  selector: 'app-root',
  templateUrl: './auto.component.html',
  styleUrls: ['./auto.component.scss']
})
export class AutoComponent {
  title = 'hackathonPortal';
  constructor(private formBuilder: FormBuilder, private hackathonService: HackathonService,
    private router: Router, private modalService: BsModalService) { }
  form: FormGroup;
  hackathon: Hackathon = new Hackathon();
  start: number = 1;
  bsModalRef: BsModalRef;
  options = {
    floor: 0,
    ceil: 100
  };
  timeOptions = {
    floor: 0,
    ceil: 24
  };
  predictResult: string;
  ngOnInit() {
    this.form = this.formBuilder.group({
      time: ['1'],
      cpu: ['1', Validators.required],
      network: ['1', Validators.required]
    });
  }

  predict() {
    this.hackathonService.setParam(this.hackathon)
      .subscribe(data => {
        if (parseInt(data) == 1) {
          this.predictResult = "Normal";
        } else {
          this.predictResult = "Anomaly";
        }
      });
  }

  train() {
    this.i = 0;
    if (this.start == 1) {
      this.hackathonService.autoTrain()
        .subscribe(data => {
          alert("training complete!");
          this.changepic();
          this.start++;
        });
    } else {
      this.changepic();
    }
  }

  t = new Array("http://100.98.136.15:5000/static/plot1.jpg", "http://100.98.136.15:5000/static/plot2.jpg", "http://100.98.136.15:5000/static/plot3.jpg", "http://100.98.136.15:5000/static/plot4.jpg",
    "http://100.98.136.15:5000/static/plot5.jpg", "http://100.98.136.15:5000/static/plot6.jpg", "http://100.98.136.15:5000/static/plot7.jpg", "http://100.98.136.15:5000/static/plot8.jpg",
    "http://100.98.136.15:5000/static/plot9.jpg", "http://100.98.136.15:5000/static/plot10.jpg", "http://100.98.136.15:5000/static/plot11.jpg", "http://100.98.136.15:5000/static/plot12.jpg",
    "http://100.98.136.15:5000/static/plot13.jpg", "http://100.98.136.15:5000/static/plot14.jpg", "http://100.98.136.15:5000/static/plot15.jpg", "http://100.98.136.15:5000/static/plot16.jpg", );
  i = 0;
  imgUrl = "../assets/picture/hackathon.jpg";
  testResultImgUrl = "";
  flag: number = 1;
  //循环方法
  changepic() {
    this.imgUrl = this.t[this.i] + "?" + new Date().getTime();
    // if (this.flag == 1 && this.i == 15) {
    //   this.flag = 0;
    // } else if (this.flag == 1 && this.i < 15) {
    //   this.flag = 1;
    // } else if (this.flag == 0 && this.i > 0) {
    //   this.flag = 0;
    // } else if (this.flag == 0 && this.i == 0) {
    //   this.flag = 1;
    // }

    // if (this.flag == 0) {
    //   this.i--;
    // } else {
    //   this.i++;
    // }
    if (this.i < 15) {
      setTimeout(() => {
        this.changepic()
        this.i++;
      }, 200);
    }
  }

  goToBasic() {
    this.router.navigate(['/auto']);
  }

  openModal(template: TemplateRef<any>) {
    this.testResultImgUrl = "http://100.98.136.15:5000/static/plot0.jpg?" + new Date().getTime();
    this.bsModalRef = this.modalService.show(template);
  }
}
