import { Router } from '@angular/router';
import { Component, OnInit, Input, Output, EventEmitter, ElementRef, ViewChild, AfterViewChecked } from '@angular/core';
import { Title } from '@angular/platform-browser';
import { FormGroup, FormBuilder, Validators, FormControl } from '@angular/forms';

import { isEmpty, kebabCase } from 'lodash';
import { isFunction } from 'lodash';
import { ModalModule } from 'ngx-bootstrap';
import { BsModalService } from 'ngx-bootstrap/modal';
import { BsModalRef } from 'ngx-bootstrap/modal/bs-modal-ref.service';
 
@Component({
    selector: 'modal-content',
    template: `
      <div class="modal-header">
        <h4 class="modal-title pull-left">{{title}}</h4>
        <button type="button" class="close pull-right" aria-label="Close" (click)="bsModalRef.hide()">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <ul *ngIf="list.length">
          <li *ngFor="let item of list">{{item}}</li>
        </ul>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-default" (click)="bsModalRef.hide()">{{closeBtnName}}</button>
      </div>
    `
})
export class TestResultComponent implements OnInit {
    title: string;
    closeBtnName: string;
    list: any[] = [];

    constructor(public bsModalRef: BsModalRef) { }

    ngOnInit() {
        this.list.push('PROFIT!!!');
    }
}
