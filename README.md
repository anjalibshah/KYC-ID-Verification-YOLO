# YOLO KYC ID Verification App


This app shows how to serve a Keras model in a Python Cloud Foundry
app. This uses the Cloud Foundry offering provided by
[IBM Cloud](https://console.ng.bluemix.net/registration/). The model file and labels
file should be put in the `assets` directory. Currently the files there correspond
to photo ID documents from which faces can be detected by retraining  *YOLOv2*. More information on YOLO can be gathered from the [original YOLO paper](https://arxiv.org/abs/1506.02640), [YOLO darknet implementation](https://pjreddie.com/darknet/yolo/), [YOLO to Keras model conversion](https://github.com/allanzelener/YAD2K), and [deep learning.ai on Coursera](https://www.deeplearning.ai) 


## 1. Clone the repo and run the app locally

Install the dependencies listed in the [requirements.txt](https://pip.readthedocs.io/en/stable/user_guide/#requirements-files) file to be able to run the app locally.

You can optionally use a
[virtual environment](https://packaging.python.org/installing/#creating-and-using-virtual-environments)
to avoid having these dependencies clash with those of other Python projects or your operating system.
  ```
pip install -r requirements.txt
  ```

Run the app.
  ```
python app.py
  ```

 View your app at: http://localhost:8000

## 2. Prepare the app for deployment

To deploy to IBM Cloud, it can be helpful to set up a manifest.yml file. 

The manifest.yml includes basic information about your app, such as the name, how
much memory to allocate for each instance and the route. In this manifest.yml **random-route: false**
prevents generating a random route for your app and allows supplying a
host name of your choice. [Learn more...](https://console.bluemix.net/docs/manageapps/depapps.html#appmanifest)

Also, the amount of memory and disk quota your app uses is dependent on the size of your model. Feel free
to increase or decrease it as you see fit.
 ```
 applications:
 - name: KYC-ID-Verification
   random-route: false
   memory: 2048M
   disk_quota: 2048M
 ```

## 3. Deploy the app

You can use the Cloud Foundry CLI to deploy apps.

Choose your API endpoint
   ```
cf api <API-endpoint>
   ```

Replace the *API-endpoint* in the command with an API endpoint from the following list.

|URL                             |Region          |
|:-------------------------------|:---------------|
| https://api.ng.bluemix.net     | US South       |
| https://api.eu-de.bluemix.net  | Germany        |
| https://api.eu-gb.bluemix.net  | United Kingdom |
| https://api.au-syd.bluemix.net | Sydney         |

Login to your IBM Cloud account

  ```
cf login
  ```

From within the *tensorflow-cf-app* directory push your app to IBM Cloud.
  ```
cf push
  ```

This can take a minute. If there is an error in the deployment process you can use the command `cf logs <Your-App-Name> --recent` to troubleshoot.

When deployment completes you should see a message indicating that your app is running.  View your app at the URL listed in the output of the push command.  You can also issue the
  ```
cf apps
  ```
  command to view your apps status and see the URL.
