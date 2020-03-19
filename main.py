from flask import Flask, request, render_template

import numpy as np
import pandas as pd

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from covid import Covid


app = Flask(__name__)
app.debug = True

@app.route("/")
def default_plot():

    data = Covid()
    data.provinces = [ 'Ontario','Quebec','Alberta', 'British Columbia']

    my_map = data.plot_provinces(data.provinces, t_norm=10, exp_fit=True)
    return render_template("index.html", plot=my_map)



if __name__ == "__main__":
    app.run(debug=True)
