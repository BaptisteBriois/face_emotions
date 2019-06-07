/* eslint-disable react/button-has-type */
/* eslint-disable no-unused-vars */
/* eslint-disable no-console */
import React, { Component } from 'react';
import ReactDOM from 'react-dom';

// import PropTypes from 'prop-types';
// import Webcam from './react-webcam';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      screenshot: null,
      tab: 0,
    };
  }

  getBase64 = (file, cb) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      cb(reader.result);
    };
    reader.onerror = (error) => {
      console.log('Error: ', error);
    };
  }

 handleClick = () => {
   const screenshot = this.webcam.getScreenshot();
   let encodedScreenshot = '';
   this.getBase64(screenshot, (result) => {
     encodedScreenshot = result;
   });

   this.setState({ screenshot });

   fetch('http://localhost:9000', {
     method: 'POST',
     headers: {
       Accept: 'application/json',
       'Content-Type': 'application/json',
     },
     body: JSON.stringify({
       encodedScreenshot,
     }),
   });
 }

 render() {
   return (
     <div>
       <h1>Face emotions</h1>
       <Webcam
         audio={false}
         ref={node => this.webcam = node}
       />
       <div>
         <h2>Screenshots</h2>
         <div className="screenshots">
           <div className="controls">
             <button onClick={this.handleClick}>Capture</button>
           </div>
           {this.state.screenshot ? <img src={this.state.screenshot} /> : null}
         </div>
       </div>
     </div>
   );
 }
}

ReactDOM.render(<App />, document.getElementById('root'));
