#!/usr/bin/env python
# coding: utf-8

# In[9]:


from flask import Flask


# In[10]:


app = Flask(__name__)


# In[11]:


from flask import request, render_template
import joblib

@app.route("/", methods = ["GET", "POST"])
def index(): 
    if request.method == "POST":
        purchases = request.form.get("purchases")
        suppcard = request.form.get("suppcard")
        purchases = float(purchases)
        suppcard = float(suppcard)
        print(purchases,suppcard)
        
        
        model1 = joblib.load("CCU_DT")
        pred1 = model1.predict([[purchases,suppcard]])
        s1 = "The score of credit card upgrade based on Decision Tree  model is " + str(pred1)
        
        model2 = joblib.load("CCU_REG")
        pred2 = model2.predict([[purchases,suppcard]])
        s2 = "Upgrading of credit card based on Linear Regression model is : " + str(pred2)
        
        
        model3 = joblib.load("CCU_NN")
        pred3 = model3.predict([[purchases,suppcard]])
        s3 = "Upgrading of credit card based on Neural Network model is : " + str(pred3)
        
        model4 = joblib.load("CCU_RF")
        pred4 = model4.predict([[purchases,suppcard]])
        s4 = "Upgrading of credit card based on Forest Tree model is : " + str(pred4)
        
        
        model5 = joblib.load("CCU_GB")
        pred5 = model5.predict([[purchases,suppcard]])
        s5 = "Upgrading of credit card based on Gradient Boosting model is : " + str(pred5)

        
        return(render_template("index.html", result1=s1, result2=s2, result3=s3, result4=s4, result5=s5))
    else: 
          return(render_template("index.html", result1="2", result2="2", result3="2", result4="2", result5="2"))
        


# In[ ]:


if __name__=="__main__":
    
    app.run()


# In[ ]:





# In[ ]:





# In[ ]:




