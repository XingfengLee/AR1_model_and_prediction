# Import the required Python libraries.
import numpy as np 
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

n       = 5 # number of days to predict

# Data from the following website:
dStr    = 'https://coronavirus.data.gov.uk'
# Use pandas to read data from csv file 'coronavirus-deaths.csv' which is downloaded from dStr.
df      = pd.read_csv('coronavirus-deaths_latest.csv',index_col=0)
data    = df.filter(like='United Kingdom', axis=0)
yVal    = data['Cumulative hospital deaths']
da      = data['Reporting date']
y       = yVal.values[::-1]
date    = da.values[::-1]
N       = len(date)

# Implement the AR(1) model using data of M length.
yOrg    = y 
y11     = np.ones((N-1,1))
for i in range(n):
	y1      = y[1+i:N+i]
	y12     = y[0:N-1]
	y13     = np.reshape(y12, (N-1, 1))
	mat     = np.hstack((y11,y13))
	inv     = np.linalg.pinv(mat)
	coef    = np.matmul(inv,y1)
	ypred   = coef*mat
	ypred1  = ypred[N-2,1]
	y       = np.hstack((y,np.round(ypred1)))
	
y_ar_pred   = y 

# Convert datetime string to datetime object.
today   = dt.date.today()
tDate   = today.strftime("%b-%d-%Y")
days    = []
for i in range(N):
	tmp = dt.datetime.strptime(date[i], '%m/%d/%Y')
	days.append(tmp)

# Generate the x-matrix of independent variables.
x       = np.arange(1,N+1)
x       = x[:, np.newaxis]
# Construct date time for the prediction. 
now     = days[N-1] 
then    = now + dt.timedelta(days=n+1)
days2   = mdates.num2date(mdates.drange(now,then,dt.timedelta(days=1) ) )
days3   = days[0:N-1] + days2 

# Plot prediction time and data.
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.plot(days3,y_ar_pred, 'bs-', label='line 1', linewidth=2)

# Plot the original data.
old       = np.arange(1,N+n+1)
y2        = np.zeros(old.shape)

y2[0:N]   = yOrg
y2[N:N+n] = None
plt.plot(days3,y2, 'ro-', label='line 1', linewidth=2)
plt.grid()
plt.ylabel("Total Number of Cumulative Hospital Deaths (Y)", fontsize=10)
plt.title('UK COVID-19 Number of Hospital Deaths Prediction on ' + tDate)  
plt.gcf().autofmt_xdate() # Rotate the x-axis datetime text
plt.text(days[1],y[2] + 12000,'Data from GOV web:  ' + dStr, fontsize=10)
plt.legend(('Prediction', 'Original Data'))
plt.show()

