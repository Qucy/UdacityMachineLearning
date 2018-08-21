import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core'; 
import { Ng5SliderModule } from 'ng5-slider'; 
import { ModalModule } from 'ngx-bootstrap';

import { AppComponent } from './app.component'; 
import {ReactiveFormsModule} from "@angular/forms";
import {HttpClientModule} from "@angular/common/http"; 
import {HackathonService} from "./services/hackathon.service";
import { AppRoutingModule } from './/app-routing.module';
import { AutoComponent } from './auto/auto.component'; 
import { BasicComponent } from './basic/basic.component';  

@NgModule({
  declarations: [
    AppComponent,
    AutoComponent,
    BasicComponent
  ],
  imports: [
    BrowserModule, 
    ReactiveFormsModule,
    HttpClientModule,
    Ng5SliderModule,
    AppRoutingModule,
    ModalModule.forRoot()
  ],
  providers: [HackathonService],
  bootstrap: [AppComponent]
})
export class AppModule { 
  static routes = AppRoutingModule;
}
