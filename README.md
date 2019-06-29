# FitDistributions

Fit continuous data with 89 different distributions (scipy.stats) to find which one fit the best.
Inspired and adapted from: https://stackoverflow.com/a/37616966 

## Usage
```
data = st.norm.rvs(1, 2, size=5000)
DF = FitDistributions(plot_all = True)
DF.distributions = [st.norm, st.maxwell, st.uniform]
dist_name, params = DF.fit_distribution(data, bins = 100)
```

#### Results
```
norm (0.9603151469277776, 1.997823976707984) 
```

![Screenshot from 2019-06-29 13-23-13](https://user-images.githubusercontent.com/20289509/60386843-0fe29400-9a71-11e9-9952-77558ea2b246.png)
