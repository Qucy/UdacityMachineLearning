import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { AutoComponent } from './auto/auto.component'; 
import { BasicComponent } from './basic/basic.component'; 

const ROUTES: Routes = [
    { path: '', redirectTo: 'basic', component: BasicComponent },  
    { path: 'basic', component: BasicComponent },  
    { path: 'auto', component: AutoComponent}
  ];
  
  @NgModule({
    imports: [
      RouterModule.forRoot(ROUTES, { useHash: false })
    ],
    exports: [RouterModule]
  })
  
  export class AppRoutesModule { }
  