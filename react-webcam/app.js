/* eslint-disable indent */
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
   this.setState({ screenshot });
   this.request(screenshot);
  }

  //  fetch('http://localhost:9000', {
  //    method: 'POST',
  //    headers: {
  //      Accept: 'application/json',
  //      'Content-Type': 'application/json',
  //    },
  //    body: JSON.stringify({
  //      encodedScreenshot,
  //    }),
  //  });
  request = async (screenshot) => {
    const pythonResponse = await fetch('http://localhost:9000', {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      mode: 'cors',
      cache: 'default',
      body: JSON.stringify({
        screenshot,
      }),
    });
    const pythonJson = await pythonResponse.json();
    console.log(pythonJson);

    const apiResponse = await fetch('https://dvic.devinci.fr/dgx/paints_torch/api/v1/colorizer', {
          method: 'POST',
          headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            sketch: pythonJson.sketch,
            hint: pythonJson.hint,
            opacity: 1,
          }),
        });

    const apiJson = await apiResponse.json();
    console.log(apiJson);
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
