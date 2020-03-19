
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'

class Covid:

    def __init__(self):

        self.get_data()


    def get_data(self):

        self.df_confirmed = pd.read_csv(url_confirmed)




    def get_country(self,name, combine=False, plot=False):

        sub_df = self.df_confirmed.loc[df_confirmed['Country/Region'] ==name]

        sub_df_timevect = sub_df.drop(columns=['Lat', 'Long','Country/Region'])
        sub_df_timevect = sub_df_timevect.set_index('Province/State').transpose()

        if(combine):
            sub_df_timevect = sub_df_timevect.sum(axis=1).to_frame(name='count')

        if(plot):
            sub_df_timevect.plot()

        return sub_df_timevect


    def get_province(self,name, plot=False):

        sub_df = self.df_confirmed.loc[self.df_confirmed['Province/State'] ==name]
        sub_df_timevect = sub_df.drop(columns=['Lat', 'Long','Country/Region'])
        sub_df_timevect = sub_df_timevect.set_index('Province/State').transpose()

        if(plot):
            sub_df_timevect.plot()

        return sub_df_timevect



    def norm_to_case(self,name, tvect, n_cases=1):

        tvect.rename(columns={name: 'count'}, inplace=True) # Add name to header

        # Find hundredth case
        filt_df = tvect[tvect['count'] > int(n_cases) - 1]
        select_case = filt_df.index[0]

        # Add to dataframe
        dt_vec = pd.to_datetime(tvect.index)
        select_case = pd.to_datetime(select_case)
        tvect['t_rel_to_case'] = (dt_vec - select_case).days

        return tvect



    def plot_provinces(self,provinces, t_norm=None, exp_fit=False):

    #     colors = np.random.randint(255, size=(len(provinces),3))
        colors = ['rgb(231,107,243)','rgb(255,100,100)','rgb(100,100,255)','rgb(100,255,100)']
    #     fig = plt.figure()
        fig = go.Figure()
        max_count = 0

        for i, prov in enumerate(provinces):

            color = 'rgb(colors[i,:])'

            t_vect = self.get_province(prov)

            new_max = np.max(t_vect.values) # Calculate max count between datasets
            if(new_max > max_count):
                max_count = new_max

            print(prov, new_max)

            # Norm to Nth case
            if t_norm is not None:

                if(exp_fit):
                    t_range, p,norm = self.est_curve(prov,t_norm,t_max=100)#, no_extrap)
                    plt.plot(norm['t_rel_to_case'], norm['count'],'-',t_range, p(t_range), '--')

                    # Plot the 2nd order polynomial fit
                    fig.add_trace(go.Scatter(x=t_range,
                                 y=p(t_range),
                                 mode='lines',
                                 name=prov + ' est.',
                                 line=dict(color=colors[i],
                                           width=1,#width=line_size[i]),
                                           dash='dash'),
                                 connectgaps=True,
                    ))

                    # Set new max value to use for scaling y-axis
                    new_max = np.max(p(t_range)) # Calculate max count between datasets
                    if(new_max > max_count):
                        max_count = new_max

                else:
                    norm = self.norm_to_case(prov, t_vect, t_norm)

                fig.add_trace(go.Scatter(x=norm['t_rel_to_case'].values,
                                         y=norm['count'].values,
                                         mode='lines',
                                         name=prov,
                                         line=dict(color=colors[i],
                                                   width=2),#line_size[i]),
                                         connectgaps=True,
                ))


        fig.update_xaxes(rangemode="nonnegative",
                         title_text='Days since first ' + str(t_norm) + ' cases')

        fig.update_yaxes(title_text='Number of confirmed cases (log-scale)')

        fig.update_layout(
            title_text="Confirmed cases (Provinces)",
            yaxis_type="log",
            yaxis_range=[np.log10(t_norm),np.log10(max_count+10)])

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON





    def plot_countries(self,countries, t_norm=None):

    #     fig = plt.figure()
        fig = go.Figure()
        max_count = 0

        for country in countries:

            t_vect = self.get_country(country,combine=True)

            new_max = np.max(t_vect.values) # Calculate max count between datasets
            if(new_max > max_count):
                max_count = new_max

            if t_norm is not None:

                norm = self.norm_to_case(country, t_vect, t_norm)

                fig.add_trace(go.Scatter(x=norm['t_rel_to_case'].values,
                                         y=norm['count'].values,
                                         mode='lines',
                                         name=country,
                        #                 line=dict(color=colors[i], width=line_size[i]),
                                         connectgaps=True,
                ))
    #             plt.plot(norm['t_rel_to_case'].values,norm['count'].values)

        fig.update_xaxes(rangemode="nonnegative",
                         title_text='Days since first ' + str(t_norm) + ' cases')

        fig.update_yaxes(title_text='Number of confirmed cases (log-scale)')

        fig.update_layout(
            title_text="Confirmed cases (Countries)",
            yaxis_type="log",
            yaxis_range=[np.log10(t_norm),np.log10(max_count+10000)])

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON

    def est_curve(self,prov, t_norm=None,t_max=20, start_x=0, no_extrap=False):

        '''
        start_x = 0 # Value to begin the exponential fit (normalized time values to x number of cases.)
        no_extrap # Fit only to as many values days are in the cropped dataframe (just fit, no extrapolation)
        '''
        t_vect = self.get_province(prov) # ** Try to make this a general function, instead of using get_province and get_country
        norm = self.norm_to_case(prov, t_vect, t_norm)

        # Sub_df contaning normalized time values including and after start_x:
        norm_crop = pd.DataFrame()
        norm_crop = norm.loc[(norm['t_rel_to_case'] >= start_x ) ]

        if(no_extrap): # Fit only to as many values days are in the cropped dataframe (just fit, no extrapolation)
            t_max = len(norm_crop.index)

        curve_fit = np.polyfit(norm_crop['t_rel_to_case'], norm_crop['count'], 2)
        p = np.poly1d(curve_fit)
        t_range = np.arange(0,t_max)

    #
        return t_range, p, norm_crop # norm_crop??


















    if __name__ == '__main__':
        # self.df_confirmed = pd.read_csv(url_confirmed)
        self.provinces = [ 'Ontario','Quebec','Alberta', 'British Columbia']
        self.plot_provinces(provinces, t_norm=10, exp_fit=False)

        # countries = [ 'Canada','US', 'Japan', 'Italy' , 'Korea, South', 'China']#, 'Hong Kong']#'Quebec','Alberta'
        # plot_countries(countries, t_norm=100)
