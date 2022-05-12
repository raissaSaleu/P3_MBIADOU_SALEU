############### PROJET 3 #################

import pandas as pd
import numpy as np 
import random
import itertools
from collections import Counter 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from sklearn import neighbors
from collections import Counter
from itertools import takewhile


def poucentageValeursManquantes(data, column):
    ''' 
        Calcule le pourcentage moyen de valeurs manquantes
        dans une dataframe donnée par valeur unique d'une colonne donnée
        
        Parameters
        ----------------
        data   : dataframe contenant les données                 
        column : str
                 La colonne à analyser
        
        Returns
        ---------------
        Un dataframe contenant:
            - une colonne "column"
            - une colonne "Percent Missing" : contenant le pourcentage de valeur manquante pour chaque valeur de colonne    
    '''
    
    percent_missing = data.isnull().sum(axis=1) * 100 / len(data.columns)
    
    return pd.DataFrame({column: data[column], 'Percent Missing': percent_missing})\
                        .groupby([column])\
                        .agg('mean')
#------------------------------------------


def tauxRemplissage(data):
    ''' 
        Cette fonction prépare les données qui seront affichées pour montrer le taux de valeurs 
        renseignées/non renseignées des colonnes dans un dataframe "data" 
        
        Parameters
        ----------------
        - data : dataframe contenant les données
 
        Returns
        ---------------
        Un dataframe contenant:
            - une colonne "Percent Missing" : pourcentage de données non renseignées
            - une colonne "Percent Filled" : pourcentage de données renseignées
            - une colonne "Total": 100
            
    '''    

    
    missing_percent_df = pd.DataFrame({'Percent Missing':data.isnull().sum()/len(data)*100})

    missing_percent_df['Percent Filled'] = 100 - missing_percent_df['Percent Missing']

    missing_percent_df['Total'] = 100

    percent_missing = data.isnull().sum() * 100 / len(data.columns)
    
    return missing_percent_df

#------------------------------------------

def plotTauxRemplissage(data, long, larg):
    ''' 
        Trace les proportions de valeurs remplies/manquantes pour chaque colonne
        dans la colonne de data sous forme de graphique à barres horizontales empilées.
        
        Parameters
        ----------------
        data   : un dataframe avec : 
                   - une colonne "Percent Missing" : pourcentage de données non renseignées
                   - une colonne "Percent Filled" : pourcentage de données renseignées
                   - une colonne "Total": 100
                                 
        long   : int 
                 La longueur de la figure
        
        larg   : int
                 La largeur de la figure
                                  
        
        Returns
        ---------------
        -
    '''
    
    data_to_plot = tauxRemplissage(data).sort_values("Percent Filled").reset_index()

    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50

    sns.set(style="whitegrid")

    #sns.set_palette(sns.dark_palette("purple", reverse=True))

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(long, larg))

    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR COLONNE",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    b = sns.barplot(x="Total", y="index", data=data_to_plot,label="non renseignées", color="thistle", alpha=0.3)
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    _, ylabels = plt.yticks()
    _, xlabels = plt.xticks()
    b.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot the Percent Filled values
    c = sns.barplot(x="Percent Filled", y="index", data=data_to_plot,label="renseignées", color="darkviolet")
    c.set_xticklabels(c.get_xticks(), size = TICK_SIZE)
    c.set_yticklabels(ylabels, size=TICK_SIZE)


    # Add a legend and informative axis label
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, frameon=True,
             fontsize=LEGEND_SIZE)

    ax.set(ylabel="Colonnes",xlabel="Pourcentage de valeurs (%)")

    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x)) + '%'))
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    plt.savefig('missingPercentagePerColumn.png')

    # Display the figure
    plt.show()
    
#------------------------------------------

#-----------------------------------------------------------------
# WARNING : only works if 2 doublons
# TODO : Systemize function below to work for 3 or more doublons
#-----------------------------------------------------------------

def fusionDoublonLignes(data, criteria_doublon):
    '''
         Cette fonction fusionne 2 lignes ayant le même code produit
        ----------------
        - data             : DataFrame contenant des doublons
        - criteria_doublon : la colonne concernée par les doublons
        
        Returns
        ---------------
        Une dataframe avec les doublons fusionnés 
    '''
    
    duplicates = data[data[criteria_doublon].duplicated()][criteria_doublon]
    data_duplicates = data[data[criteria_doublon].isin(duplicates)]

    #-----------------------------------------------------------------
    # Fusion des données des doublons :
    # la nouvelle ligne devient la première, avec ses valeurs NaN
    # remplacées par les valeurs non NaN de la deuxième ligne
    #-----------------------------------------------------------------

    merged_duplicates = pd.DataFrame()

    for criteria_duplicate in duplicates:
        tmp_df = data_duplicates[data_duplicates[criteria_doublon]==criteria_duplicate]

        obj_df = tmp_df.select_dtypes(include=[np.object])
        num_df = tmp_df.select_dtypes(exclude=[np.object])

        merged_row = pd.concat([obj_df.head(1).fillna(obj_df.tail(1)).reset_index(drop=True),
                                num_df.max().to_frame().T.reset_index(drop=True)], axis=1)

        merged_duplicates = pd.concat([merged_duplicates, merged_row])

    #-----------------------------------------------------------------
    # Création d'un nouveau dataframe sans les doublons, mais avec les lignes nouvellement fusionnées
    #-----------------------------------------------------------------

    return pd.concat([data[~data[criteria_doublon].isin(duplicates)],
                            merged_duplicates])
     
#------------------------------------------

def remplacerValeurs(data, column, values):
    '''
        Cette fonction remplace les valeurs d'une colonne par d'autres valeurs
        
        ----------------
        - data   : DataFrame contenant les valeurs à remplacer
        - column : Colonne contenant les valeurs à remplacer
        - values : Un dictionnaire contenant les valeurs à remplacer et les nouvelles valeurs
        
        Returns
        ---------------
        _
    '''
    
    for value_error, value_correct in values.items():
        data.loc[data[column] == value_error, column] = value_correct
        
#------------------------------------------

def remplacerValeursNaN(data, cols, criterion, valueType):
    '''
        Remplace les valeurs NaN dans une liste des colonnes données 
        regroupées selon un critère par le mode ou la moyenne 
        selon que la colonne est une variable quantitative ou qualitativz("QUANT" ou "QUAL")
        
        ----------------
        - data      : DataFrame contenant les valeurs NaN
        - cols      : Liste contenant les noms des colonnes dans lesquelles remplacer les valeurs NaN
        - criterion : Critère de regroupement des données
        - valueType : Le type des variables dans cols ("QUAL" or "QUANT")
        
        Returns
        ---------------
        _
    '''
    
    for column in cols:
        if criterion != None:
            value_per_criterion = {}

            for val_criterion, data_df in data.groupby([criterion]):
                if valueType == "QUAL":
                    value_per_criterion[val_criterion] = data_df[column].mode()[0]
                elif valueType == "QUANT":
                    value_per_criterion[val_criterion] = data_df[column].mean()

            for criterion_value, value in value_per_criterion.items():
                data.loc[data[criterion] == criterion_value, column] \
                = \
                data.loc[data[criterion] == criterion_value, column].fillna(value)
        else:
            if valueType == "QUAL":
                value = data[column].mode()[0]
            elif valueType == "QUANT":
                value = data[column].mean()
            else:
                raise Exception("Invalid value type :" + valueType)

            data[column] = data.loc[:, column].fillna(value)
    
#------------------------------------------

def remplacerNaNVarQualitative(data, qualitative_cols, criterion=None):
    '''        
        Remplace les valeurs NaN dans une liste des colonnes (qualitatives) 
        regroupées selon un critère par le mode 
        
        ----------------
        - data             : DataFrame contenant les valeurs NaN
        - qualitative_cols : Liste contenant les noms des colonnes dans lesquelles remplacer les valeurs NaN
        - criterion        : Critère de regroupement des données
        
        Returns
        ---------------
        _
    '''
    
    remplacerValeursNaN(data, qualitative_cols, criterion, "QUAL")

#------------------------------------------

def remplacerNaNVarQuantitative(data, quantitative_cols, criterion=None):
    '''        
        Remplace les valeurs NaN dans une liste des colonnes (quantitatives) 
            regroupées selon un critère par la moyenne
        
        ----------------
        - data             : DataFrame contenant les valeurs NaN
        - qualitative_cols : Liste contenant les noms des colonnes dans lesquelles remplacer les valeurs NaN
        - criterion        : Critère de regroupement des données
        
        Returns
        ---------------
        _
    '''
    
    remplacerValeursNaN(data, quantitative_cols, criterion, "QUANT")
              
#------------------------------------------

def plotBoxPlots(data, long, larg, nb_rows, nb_cols):
    '''
        Affiche un boxplot pour chaque colonne dans data.
        
        Parameters
        ----------------
        data : dataframe contenant des exclusivment des variables quantitatives
                                 
        long : int
               longueur de la figure
        
        larg : int
               largeur de la figure
               
        nb_rows : int
                  Le nombre de lignes dans le subplot
        
        nb_cols : int
                  Le nombre de colonnes dans le subplot
                                  
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 35
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 10
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("BOXPLOT DES VARIABLES QUANTITATIVES", fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    row = 0
    column = 0

    for ind_quant in data.columns.tolist():
        ax = axes[row, column]

        sns.despine(left=True)

        #b = sns.boxplot(x=np.log10(data[ind_quant]), ax=ax, color="darkviolet")
        b = sns.boxplot(x=data[ind_quant], ax=ax, color="darkviolet")


        plt.setp(axes, yticks=[])

        plt.tight_layout()

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

        lx = ax.get_xlabel()
        ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        if ind_quant == "salt_100g":
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        ly = ax.get_ylabel()
        ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.tick_params(axis='both', which='major', pad=TICK_PAD)

        ax.xaxis.grid(True)
        ax.set(ylabel="")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
                    
#------------------------------------------

def suppressionOutliers(data, quantitative_columns, criterion=None):
    '''
        Supprime les valeurs aberrantes des colonnes quantivatives dans les données regroupées selon un critère
        
        ----------------
        - data                : Dataframe contenant les données à supprimer
        - qualitative_columns : Liste de colonnes quantitatives
        - criterion            : Critère de regroupement des données
        
        Returns
        ---------------
        _ filtered_df : dataframe sans outliers
    '''
    
    filtered_df = data.copy()

    if criterion != None:
        ## TODO : IMPLEMENT
        #for criterion_value, data_criterion in filtered_df.groupby([criterion]):
        print("TO IMPLEMENT")
        
    else:
        for column in quantitative_columns:
            Q1 = filtered_df[column].quantile(0.25)
            Q3 = filtered_df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
            filtered_df = filtered_df.query('(@Q1 - 1.5 * @IQR) <= '+ str(column) +' <= (@Q3 + 1.5 * @IQR)')
          
    return filtered_df

#------------------------------------------


def plotPieChart(data, groupby_col, long, larg, title, title_fig_save):
    ''' 
        Plots a pie chart of the proportion of each modality for groupby_col
        with the dimension (long, larg), with the given title and saved figure
        title.
        
        Parameters
        ----------------
        data           : pandas dataframe containing the data, with a "groupby_col"
                         column
        
        groupby_col    : the name of the quantitative column of which the modality
                         frequency should be plotted.
                                  
        long           : int
                         The length of the figure for the plot
        
        larg           : int
                         The width of the figure for the plot
        
        title          : title for the plot
        
        title_fig_save : title under which to save the figure
                 
        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 25
    TITLE_PAD = 60

    # Initialize the figure
    f, ax = plt.subplots(figsize=(long, larg))


    # Set figure title
    plt.title(title,fontweight="bold",fontsize=TITLE_SIZE, pad=TITLE_PAD,)
       
    # Create pie chart for topics
    a = data[groupby_col].value_counts(normalize=True).plot(kind='pie',
                                                        autopct=lambda x:'{:2d}'.format(int(x)) + '%', 
                                                        fontsize =20)
    # Remove y axis label
    ax.set_ylabel('')
    
    # Make pie chart round, not elliptic
    plt.axis('equal') 
    
    # Save the figure 
    plt.savefig(title_fig_save)
    
    # Display the figure
    plt.show()

#------------------------------------------

def plotQualitativeDist(data, long, larg):
    '''
        Affiche un graphique à barres indiquant la fréquence
        des modalités pour chaque colonne de données.
        
        Parameters
        ----------------
        data : dataframe contenant des données qualitatives
                                 
        long : int
               longueur de l figure
        
        larg : int
               largeur de la figure
                                  
        
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 45
    TITLE_PAD = 1.05
    TICK_SIZE = 25
    TICK_PAD = 20
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    nb_rows = 2
    nb_cols = 2

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("DISTRIBUTION DES VARIABLES QUALITATIVES", fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    row = 0
    column = 0

    for ind_qual in data.columns.tolist():
        
        data_to_plot = data.sort_values(by=ind_qual).copy()
        
        ax = axes[row, column]
        
        if ind_qual == "nutriscore_grade":
            nutri_colors = ["#287F46", "#78BB42", "#F9C623", "#E6792B", "#DC3D2A"]
            b = sns.countplot(y=ind_qual, data=data_to_plot, palette=sns.color_palette(nutri_colors),ax=ax)
        elif ind_qual == "nova_group":
            b = sns.countplot(y=ind_qual, data=data_to_plot, palette="Purples", ax=ax)
        else:
            b = sns.countplot(y=ind_qual, data=data_to_plot,
                              color="darkviolet",
                              ax=ax,
                              order = data_to_plot[ind_qual].value_counts().index)


        plt.tight_layout()
        
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=1.4, hspace=0.2)

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        
        if ind_qual == "nova_group":
            ylabels = [item.get_text()[0] for item in ax.get_yticklabels()]
        else:
            ylabels = [item.get_text().upper() for item in ax.get_yticklabels()]
        b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")

        lx = ax.get_xlabel()
        ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        ly = ax.get_ylabel()
        ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))
        
        ax.xaxis.grid(True)

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
        
#------------------------------------------

def calculeLorenzGini(data):
    '''
        Calcule la courbe de Lorenz et le coefficient de Gini pour une variable donnée
        
        ----------------
        - data       : data series
        
        Returns
        ---------------
        Un tuple contenant :
        - lorenz_df : une liste contenant les valeurs de la courbe de Lorenz
        - gini_coeff : les coefficients de Gini associés
        
        Source : www.openclassrooms.com
    '''
    
    dep = data.dropna().values
    n = len(dep)
    lorenz = np.cumsum(np.sort(dep)) / dep.sum()
    lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

    #---------------------------------------------------
    # Gini :
    # Surface sous la courbe de Lorenz. Le 1er segment
    # (lorenz[0]) est à moitié en dessous de 0, on le
    # coupe donc en 2, on fait de même pour le dernier
    # segment lorenz[-1] qui est à 1/2 au dessus de 1.
    #---------------------------------------------------

    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
    # surface entre la première bissectrice et le courbe de Lorenz
    S = 0.5 - AUC
    gini_coeff = [2*S]
         
    return (lorenz, gini_coeff)
    
#------------------------------------------

def calculeLorenzsGinis(data):
    '''
        Calcule la courbe de Lorenz et les coefficients de Gini
        pour toutes les colonnes d'une dataframe
        
        ----------------
        - data       : dataframe
        
        Returns
        ---------------
        Un tuple contenant :
        - lorenz_df : une dataframne contenant les valeurs de la courbe de Lorenz
                      pour chaque colonne de la dataframe
        - gini_coeff : une dataframe contenant le coeff de Gini associé pour
                       chaque colonne de la dataframe
    '''
    
    ginis_df = pd.DataFrame()
    lorenzs_df = pd.DataFrame()

    for ind_quant in data.columns.unique().tolist():
        lorenz, gini = calculeLorenzGini(data[ind_quant])
        ginis_df[ind_quant] = gini
        lorenzs_df[ind_quant] = lorenz

    n = len(lorenzs_df)
    xaxis = np.linspace(0-1/n,1+1/n,n+1)
    lorenzs_df["index"]=xaxis[:-1]
    lorenzs_df.set_index("index", inplace=True)
    
    ginis_df = ginis_df.T.rename(columns={0:'Indice Gini'})
    
    return (lorenzs_df, ginis_df)

#------------------------------------------

def plotLorenz(lorenz_df, long, larg):
    '''
        Dessin la courbe de Lorenz
        
        ----------------
        - lorenz_df : une dataframe contenant les valeurs de Lorenz
                      une colonne = valeur Lorenz pour une variable
        - long       : int
                       longueur de  la figure
        
        - larg       : int
                       largeur de la figure
        
        Returns
        ---------------
        _
    '''
    
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50


    sns.set(style="whitegrid")
    
    f, ax = plt.subplots(figsize=(long, larg))
    
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    plt.title("VARIABLES QUANTITATIVES - COURBES DE LORENZ",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    sns.set_color_codes("pastel")
    
    b = sns.lineplot(data=lorenz_df, palette="pastel", linewidth=5, dashes=False)
    
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

    b.set_yticklabels(b.get_yticks(), size = TICK_SIZE)

    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
    
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    ax.set_xlabel("")

    # Add a legend and informative axis label
    leg = ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, frameon=True,
             fontsize=LEGEND_SIZE)
    
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)


    # Display the figure
    plt.show()

#------------------------------------------

def plotDistplotsRug(data, long, larg, nb_rows, nb_cols):
    '''
        Trace la distribution de toutes les colonnes dans data
        (doit être des colonnes quantitatives uniquement)
        couplée à un graphique en tapis de la distribution
        
        Parameters
        ----------------
        data : dataframe contenant exclusivement des données quatitatives
                                 
        long : int
               longueur de la figure
        
        larg : int
               largeur de la figure
               
        nb_rows : int
                  nombre de lignes du subplot
        
        nb_cols : int
                  nombre de colonnes du subplot
                                 
        Returns
        ---------------
        -
    '''
        
    TITLE_SIZE = 30
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    sns.set_palette(sns.dark_palette("purple", reverse="True"))

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("DISTRIBUTION DES VARIABLES QUANTITATIVES", fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = 0
    column = 0

    for ind_quant in data.columns.tolist():

        sns.despine(left=True)

        ax = axes[row, column]

        b = sns.distplot(data[ind_quant], ax=ax, rug=True)

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        b.set_xlabel(ind_quant,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        if ind_quant in ["saturated-fat_100g", "salt_100g"]:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        plt.setp(axes, yticks=[])

        plt.tight_layout()

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

#------------------------------------------

def plotPairplot(data, height, hue=None):
    '''
        Pairplot des variables quantitatives dans data
        
        ----------------
        - data   : dataframe contenant les données
        - height : hauteur de la figure
        - hue    : S'il est donné, le tracé sera colorié en fonction des
                   valeurs de cette variables
        
        Returns
        ---------------
        _
    '''
    TITLE_SIZE = 70
    TITLE_PAD = 1.05
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 45
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5
            
    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title

    test_palette = {"a":"#287F46", "b":"#78BB42", "c":"#F9C623", "d":"#E6792B", "e":"#DC3D2A"}

    plt.rcParams["font.weight"] = "bold"

    if hue == None:
        b = sns.pairplot(data=data, height=height)
    else:
        b = sns.pairplot(data=data, hue=hue, height=height, palette=test_palette)

    b.fig.suptitle("VARIABLES QUALITATIVES - PAIRPLOT",
                   fontweight="bold",
                   fontsize=TITLE_SIZE, y=TITLE_PAD)



#------------------------------------------

def plotCorrelationHeatMap(data, corr_method, long, larg):
    '''
        heatmap des coefficients de corrélation entre les colonnes quantitatives
        
        ----------------
        - data : une dataframe contenant les données
        - corr : méthode de correlation ("pearson" or "spearman")
        - long : int
                 longueur de la figure
        
        - larg : int
                 largeur de la figure 
        
        Returns
        ---------------
        _
    '''
    
    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 45
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5
    
    corr = data.corr(method = corr_method)

    f, ax = plt.subplots(figsize=(long, larg))
                
    f.suptitle("COEFFICIENT DE CORRÉLATION DE " + corr_method.upper(), fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax,
                    annot=corr, annot_kws={"fontsize":20}, fmt=".2f")

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    b.set_xticklabels(xlabels, size=TICK_SIZE, weight="bold")
    b.set_xlabel(data.columns.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    b.set_ylabel(data.index.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()
                
#------------------------------------------

def plotTableContingenceKhi2(data, X, Y, long, larg):
    '''
        Affiche une table de contingence de var1 et var2 coloré en 
        fonction du test Khi2
        
        ----------------
        - data : une dataframe contenant les données
        - X,Y : 2 variables qualitatives
        - long : int
                 longueur de la figure
        
        - larg : int
                 largeur de la figure 
        
        Returns
        ---------------
        _
    '''
    
    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 18
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    cont = data[[X,Y]].pivot_table(index=Y,columns=X,aggfunc=len,margins=True,margins_name="Total").fillna(0).copy().astype(int)
    
    
    stat, p, dof, expected = chi2_contingency(cont)

    measure = (cont-expected)**2/expected
    table = measure/stat
    
    
    #tx = cont.loc[:,["Total"]]
    #ty = cont.loc[["Total"],:]
    #n = len(data)
    #indep = tx.dot(ty) / n

    #c = cont.fillna(0) # On remplace les valeurs nulles par 0
    #measure = (c-indep)**2/indep
    #xi_n = measure.sum().sum()
    #table = measure/xi_n

    f, axes = plt.subplots(figsize=(long, larg))

    f.suptitle("TABLEAU DE CONTINGENCE\nAVEC MISE EN LUMIÈRE DES RELATIONS PROBABLES (KHI-2)", fontweight="bold",
                   fontsize=TITLE_SIZE, y=TITLE_PAD)

    b=sns.heatmap(table.iloc[:-1,:-1],annot=cont.iloc[:-1,:-1],annot_kws={"fontsize":20}, fmt="d")

    xlabels = [item.get_text().upper() for item in axes.get_xticklabels()]
    b.set_xticklabels(xlabels, size=TICK_SIZE, weight="bold")
    b.set_xlabel(table.iloc[:-1, :-1].columns.name.upper(),fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ylabels = [item.get_text() for item in axes.get_yticklabels()]
    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    b.set_ylabel(table.iloc[:-1, :-1].index.name.upper(),fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()
    
#------------------------------------------

def testKhi2(data, X, Y):
    '''
        Test Khi2
        
        ----------------
        - data : une dataframe contenant les données
        - X,Y : 2 variables qualitatives
        
        Returns
        ---------------
        - stat         : the result of the Khi-2 test
        - p            : the p-value
        - dof          : degrees of freedom
    '''
    cont = data[[X,Y]].pivot_table(index=Y,columns=X,aggfunc=len,margins=True,margins_name="Total").fillna(0).copy().astype(int)
    stat, p, dof, expected = chi2_contingency(cont)
    
    return (stat, p, dof)
    
#------------------------------------------

def areDependent_withStatTest(stat, dof, prob=0.95):
    '''
        Checks independence of variables using Khi-2
        test value
        
        ----------------
        - stat : The Khi-2
        - dof  : degrees of freedom
        - prob : probability to be right rejecting H0
        
        Returns
        ---------------
        True  : H0 can be rejected with 1-prob chances
                to be wrong
        False : H0 could not be rejected for the prob
                given
    '''
    
    return abs(stat) >= chi2.ppf(prob, dof)

#------------------------------------------

def areDependent_withPValue(p, prob=0.95):
    '''
        Checks independence of variables using p-value
        from Khi-2 test
        
        ----------------
        - p : p-value from Khi-2 test
        - prob : probability to be wrong rejecting H0
        
        Returns
        ---------------
        True  : H0 can be rejected with prob chances
                to be right
        False : H0 could not be rejected for the prob
                given
    '''
    
    return p < 1.0 - prob
    
#------------------------------------------

def testDependance(stat, dof, p_value, prob=0.95):
    '''
        Checks independence of variables examining
        Khi-2 and p-values.
        
        ----------------
        - stat : The Khi-2
        - dof  : degrees of freedom
        - p_value : p-value
        - prob : probability to be right rejecting H0
        
        Returns
        ---------------
        True  : H0 can be rejected with 1-prob chances
                to be wrong
        False : H0 could not be rejected for the prob
                given
    '''
    
    if areDependent_withStatTest(stat, dof) and areDependent_withPValue(p_value):
        print("Les variables sont dépandantes - H0 rejetée")
    elif ~areDependent_withStatTest(stat, dof):
        print("Le test statistique n'a pas réussi à rejeter H0 - les variables peuvent être indépendantes")
    elif ~areDependent_withPValue(p_value):
        print("Le test de la p-value n'a pas réussi à rejeter H0 - les variables peuvent être indépendantes")
    else:
        print("Les variables sont indépendantes - Ne pas rejeter H0")

#------------------------------------------

def plotQualQuantDist(data, X, Y, long, larg):
    '''
        Affiche un graphique à barres indiquant la fréquence
        des modalités pour chaque colonne de données.
        
        Parameters
        ----------------
        data : dataframe contenant des données qualitatives
        
        X    : variable qualitative
        
        Y    : variable quantitative
                                 
        long : int
               longueur de la figure
        
        larg : int
               largeur de la figure
                                  
        
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 25
    TITLE_PAD = 1
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 10
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    
    sous_echantillon = data.copy()

    modalites = sous_echantillon[X].sort_values(ascending=True).unique()
    groupes = []
    for m in modalites:
        groupes.append(sous_echantillon[sous_echantillon[X]==m][Y])
    
    # Propriétés graphiques (pas très importantes)    
    medianprops = {'color':"black"}
    meanprops = {'marker':'o', 'markeredgecolor':'black',
                'markerfacecolor':'firebrick'}

    f, axes = plt.subplots(figsize=(long, larg))

    f.suptitle("BOXPLOT "+ X +"/"+Y, fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    b = plt.boxplot(groupes, labels=modalites, showfliers=False, medianprops=medianprops, 
                vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
    axes.set_xlabel(Y.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    axes.set_ylabel(X.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    #if X == "nutriscore_grade" :
    #    ylabels = [item.get_text().upper() for item in axes.get_yticklabels()]
    #    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    plt.show()
    
#------------------------------------------

def eta_squared(data, x_qualit,y_quantit):
    '''
        Calculate the proportion of variance
        in the given quantitative variable for
        the given qualitative variable
        
        ----------------
        - data      : The dataframe containing the data
        - x_quantit : The name of the qualitative variable
        - y_quantit : The name of the quantitative variable
        
        Returns
        ---------------
        Eta_squared
    '''
    
    sous_echantillon = data.copy().dropna(how="any")

    x = sous_echantillon[x_qualit]
    y = sous_echantillon[y_quantit]

    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
#------------------------------------------

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(8,8))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                      plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            an = np.linspace(0, 2 * np.pi, 100)  # Add a unit circle for scale
            plt.plot(np.cos(an), np.sin(an))
            plt.axis('equal')

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)) , fontsize=18)
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)), fontsize=18)

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1), fontsize=18)
            plt.show(block=False)
            
#------------------------------------------

