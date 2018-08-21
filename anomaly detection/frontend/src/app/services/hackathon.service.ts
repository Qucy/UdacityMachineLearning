import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import {Hackathon} from "../model/Hackathon.model";
import { Observable, Subject, of } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';

@Injectable()
export class HackathonService {
  constructor(private http: HttpClient) { }
  baseUrl: string = 'http://100.98.136.15:5000';

  getDatas() { 
    return this.http.get<Hackathon[]>(this.baseUrl);
  }

  getDataById(id: number) {
    return this.http.get<Hackathon>(this.baseUrl + '/' + id);
  }

  setParam(hackathon: Hackathon) { 
    return this.http.post(this.baseUrl+ '/predict', hackathon, {responseType: 'text'});
  }  

  basicTrain():Observable<string>{
    let model = new Hackathon();
    model.cpu = 10;
    model.network = 10;
    model.time = -1;
    let a = this.http.post(this.baseUrl + '/train1', model, {responseType: 'text'});
    return a;
  }

  private handleError<T> (operation = 'operation', result?: T) {
    return (error: any): Observable<T> => {
 
      // TODO: send the error to remote logging infrastructure
      console.error(error); // log to console instead
 
      // TODO: better job of transforming error for user consumption
      this.log(`${operation} failed: ${error.message}`);
 
      // Let the app keep running by returning an empty result.
      return of(result as T);
    };
  }

  private log(message: string) {
    console.log(message);
  }

  benchmark():Observable<any>{
    return this.http.post(this.baseUrl + '/benchmark', "", {responseType: 'text'});
  }

  autoTrain(){
    let model = new Hackathon();
    model.cpu = 10;
    model.network = 10;
    model.time = -1;
    return this.http.post(this.baseUrl + '/train2', model, {responseType: 'text'});    
  }
}
