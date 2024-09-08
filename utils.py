import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import shap


def get_data(path):
    # Read the data from the csv file and preprocess it
    df = pd.read_csv(path)
    df['pain improvement percentage'] = df['pain improvement'] / df['Baseline pain score']
    df['co_morbidities'] = df['co_morbidities'].replace({'No': -1, 'Yes': 1})
    df['Opioids consumption'] = df['Opioids consumption'].replace({'No': -1, 'Yes': 1})
    df['sport'] = df['sport'].replace({'No': -1, 'Yes': 1})
    df['PreLicenseUsage'] = df['PreLicenseUsage'].replace({'No': -1, 'Yes': 1})
    df['Smoke Tobacco'] = df['Smoke Tobacco'].replace({'No': -1, 'Yes': 1})
    df['Alcohol'] = df['Alcohol'].replace({'No': -1, 'Yes': 1})
    df['sleep_duration'] = df['sleep_duration'].replace({'disturbed': -1, 'normative': 1})
    df["Gender"] = df["Gender"].replace({'Male': -1, 'Female': 1})

    return df

def transform_column(value, threshold=0.25, percentage=True):
    """
    :param value: column value
    :param threshold: percentage threshold for treatment success
    :return: change the value to 1 or 0 acording to treatment success criteria
    """
    if percentage:
        return 1 if value >= threshold else 0
    else:
        return 1 if value > 1 else 0


def get_columns_names():
    dem_clinic_features = ['pain improvement percentage', 'Opioids consumption','Baseline pain score' , 'sport', 'co_morbidities',
                            'BDI score', 'PDI score', 'PreLicenseUsage', 'Smoke Tobacco', 'Alcohol',
                           'BMI', 'headaches','Neuropathic pain','musculoskeletal pain','dysfunctional pain','visceral pain', 'sleep_duration', 'Gender', 'Age']

    left = ['pain improvement percentage', "Total THC", "Total CBD", "Total CBG", 'α-Pinene','β-Citronellene', 'Camphene', 'Sabinene', 'β-Pinene', 'β-Myrcene', '2-Carene', '3Δ-Carene', 'α-Phellandrene', 'α-Terpinene', '1,4-Cineole', 'Limonene', 'cis-Ocimene', 'm-Cymene', 'Eucalyptol', 'p-Cymene', 'trans-Ocimene', 'γ-Terpinene', 'Sabynene hydrate', 'cis-Linalool oxide', 'Terpinolene', 'trans-Linalool oxide', 'Linalool', 'α-Pinene oxide', 'Fenchone', 'α-Fenchol', 'α-Thujone', 'β-Thujone', 'cis-Limonene oxide', 'trans-Pinocarveol', 'trans-Limonene oxide', 'Isopulegol', 'Citronellal', 'Isoborneol', 'Menthol', 'Borneol', 'Camphor', 'Terpinen-4-ol', 'Isomenthol', 'α-Terpineol', 'Citronellol', 'Nerol', '4-Allylanisole', 'cis-Dihydrocarvone', 'Linalyl acetate', 'Myrtenal', 'cis-Carveol', 'trans-Dihydrocarvone', 'β-Cyclocitral', 'trans-Carveol', 'Geraniol', 'Verbenone', 'cis-Citral', 'α-Pulegone', 'Menthyl acetate', 'Bornyl acetate', 'Isobornyl acetate', 'Cuminaldehyde', 'Carvone', 'Piperitone', 'trans-Citral', 'α-Longipinene', 'Thymol', 'Carvacrol', 'Perillyl alcohol', 'Citronellyl acetate', 'α-Funebrene', 'β-Elemene', 'trans-Terpin', 'α-Gurjunene', 'Isolongifolene', 'Longifolene', 'β-Funebrene', 'α-Cedrene', 'β-Caryophyllene', 'Azulene', 'Geranyl acetate', 'β-Cedrene', 'Thujopsene', 'Aromadendrene', 'Carvacryl acetate', 'Eugenol', 'trans-β-Farnesene', 'α-Humulene', 'Alloaromadendrene', 'β-Chamegrene', 'Ledene', 'α-Curcumene', 'Valencene', 'β-Curcumene', 'β-Ionone', 'Cuparene', 'trans-Nerolidol', 'Guaiol', 'Caryophylllene oxide', 'Cedrol', 'β-Eudesmol', 'α-Bisabolol', 'trans-Farnesol']
    left = [col for col in left if col not in ['Citronellol', 'cis-Carveol', 'trans-Carveol', 'β-Chamegrene', 'β-Eudesmol', 'uv_CBL', 'uv_d8-THC', 'uv_CBCV', 'uv_Cannabicitran','MS_CBG-C4','MS_CBGO','MS_CBGM','MS_THCO','MS_THCM','MS_CBNO','MS_CBNA-8-OH','MS_CBNM','MS_CBNDVA']]

    right = ['pain improvement percentage', 'MS_CBGA', 'CBG', 'MS_CBGA-C4', 'MS_CBG-C4', 'MS_CBGVA', 'MS_CBGV', 'MS_CBGOA', 'MS_CBGO', 'MS_CBGMA', 'MS_CBGM', 'MS_Sesqui-CBGA', 'MS_Sesqui-CBG', 'MS_THCA', 'MS_THC', 'MS_THCA-C4', 'MS_THC-C4', 'MS_THCVA', 'MS_THCV', 'MS_THCOA', 'MS_THCO', 'MS_THCMA', 'MS_THCM', 'MS_CBDA', 'MS_CBD', 'MS_CBDA-C4', 'MS_CBD-C4', 'MS_CBDVA', 'MS_CBDV', 'MS_CBDOA', 'MS_CBDO', 'MS_CBDMA', 'MS_CBDM', 'MS_CBCA', 'MS_CBC', 'MS_CBCA-C4', 'MS_CBC-C4', 'MS_CBCVA', 'MS_CBCV', 'MS_CBCOA', 'MS_CBCO', 'CBNA', 'MS_CBN', 'MS_CBNA-C4', 'MS_CBN-C4', 'MS_CBNVA', 'MS_CBNV', 'MS_CBNOA', 'MS_CBNO', 'MS_CBNA-8-OH', 'MS_CBN-8-OH', 'MS_CBNM', 'MS_CBEA', 'MS_CBE', 'MS_CBEVA', 'MS_CBEV', 'MS_CBNDA', 'MS_CBND', 'MS_CBNDVA', 'MS_d8-THC', 'MS_CBL', 'MS_CBTA-1', 'CBT-1', 'MS_CBTV-1', 'MS_CBTA-3', 'MS_CBT-3', 'MS_CBTV-3', 'MS_CBT-2', 'MS_329-11a', 'MS_329-11b', 'MS_329-11c', 'MS_329-11d', 'MS_329-11e', 'MS_373-12a', 'MS_373-12b', 'MS_373-12c', 'MS_373-12d', 'MS_327-13a', 'MS_327-13b', 'MS_327-13c', 'MS_371-14a', 'MS_371-14b', 'MS_417-15a', 'MS_373-15b', 'MS_373-15c', 'MS_357-16a', 'MS_313-16b', 'MS_361-17a', 'MS_361-17b', 'MS_331-18a', 'MS_331-18b', 'MS_331-18c', 'MS_331-18d', 'MS_375-19a', 'MS_375-19b', 'MS_375-19c']
    right = [col for col in right if col not in ['Citronellol', 'cis-Carveol', 'trans-Carveol', 'β-Chamegrene', 'β-Eudesmol', 'uv_CBL', 'uv_d8-THC', 'uv_CBCV', 'uv_Cannabicitran','MS_CBG-C4','MS_CBGO','MS_CBGM','MS_THCO','MS_THCM','MS_CBNO','MS_CBNA-8-OH','MS_CBNM','MS_CBNDVA']]

    all_features = ['pain improvement percentage', 'Opioids consumption','Baseline pain score' , 'sport', 'co_morbidities',
                            'BDI score', 'PDI score', 'PreLicenseUsage', 'Smoke Tobacco', 'Alcohol',
                           'BMI', 'headaches','Neuropathic pain','musculoskeletal pain','dysfunctional pain','visceral pain', 'sleep_duration', 'Gender', 'Age',"Total THC", "Total CBD", "Total CBG", 'α-Pinene','β-Citronellene', 'Camphene', 'Sabinene', 'β-Pinene', 'β-Myrcene', '2-Carene', '3Δ-Carene', 'α-Phellandrene', 'α-Terpinene', '1,4-Cineole', 'Limonene', 'cis-Ocimene', 'm-Cymene', 'Eucalyptol', 'p-Cymene', 'trans-Ocimene', 'γ-Terpinene', 'Sabynene hydrate', 'cis-Linalool oxide', 'Terpinolene', 'trans-Linalool oxide', 'Linalool', 'α-Pinene oxide', 'Fenchone', 'α-Fenchol', 'α-Thujone', 'β-Thujone', 'cis-Limonene oxide', 'trans-Pinocarveol', 'trans-Limonene oxide', 'Isopulegol', 'Citronellal', 'Isoborneol', 'Menthol', 'Borneol', 'Camphor', 'Terpinen-4-ol', 'Isomenthol', 'α-Terpineol', 'Citronellol', 'Nerol', '4-Allylanisole', 'cis-Dihydrocarvone', 'Linalyl acetate', 'Myrtenal', 'cis-Carveol', 'trans-Dihydrocarvone', 'β-Cyclocitral', 'trans-Carveol', 'Geraniol', 'Verbenone', 'cis-Citral', 'α-Pulegone', 'Menthyl acetate', 'Bornyl acetate', 'Isobornyl acetate', 'Cuminaldehyde', 'Carvone', 'Piperitone', 'trans-Citral', 'α-Longipinene', 'Thymol', 'Carvacrol', 'Perillyl alcohol', 'Citronellyl acetate', 'α-Funebrene', 'β-Elemene', 'trans-Terpin', 'α-Gurjunene', 'Isolongifolene', 'Longifolene', 'β-Funebrene', 'α-Cedrene', 'β-Caryophyllene', 'Azulene', 'Geranyl acetate', 'β-Cedrene', 'Thujopsene', 'Aromadendrene', 'Carvacryl acetate', 'Eugenol', 'trans-β-Farnesene', 'α-Humulene', 'Alloaromadendrene', 'β-Chamegrene', 'Ledene', 'α-Curcumene', 'Valencene', 'β-Curcumene', 'β-Ionone', 'Cuparene', 'trans-Nerolidol', 'Guaiol', 'Caryophylllene oxide', 'Cedrol', 'β-Eudesmol', 'α-Bisabolol', 'trans-Farnesol', 'MS_CBGA', 'CBG', 'MS_CBGA-C4', 'MS_CBGVA', 'MS_CBGV', 'MS_CBGOA', 'MS_CBGO', 'MS_CBGMA', 'MS_CBGM', 'MS_Sesqui-CBGA', 'MS_Sesqui-CBG', 'MS_THCA', 'MS_THC', 'MS_THCA-C4', 'MS_THC-C4', 'MS_THCVA', 'MS_THCV', 'MS_THCOA', 'MS_THCO', 'MS_THCMA', 'MS_THCM', 'MS_CBDA', 'MS_CBD', 'MS_CBDA-C4', 'MS_CBD-C4', 'MS_CBDVA', 'MS_CBDV', 'MS_CBDOA', 'MS_CBDO', 'MS_CBDMA', 'MS_CBDM', 'MS_CBCA', 'MS_CBC', 'MS_CBCA-C4', 'MS_CBC-C4', 'MS_CBCVA', 'MS_CBCV', 'MS_CBCOA', 'MS_CBCO', 'CBNA', 'MS_CBN', 'MS_CBNA-C4', 'MS_CBN-C4', 'MS_CBNVA', 'MS_CBNV', 'MS_CBNOA', 'MS_CBNO', 'MS_CBNA-8-OH', 'MS_CBN-8-OH', 'MS_CBNM', 'MS_CBEA', 'MS_CBE', 'MS_CBEVA', 'MS_CBEV', 'MS_CBNDA', 'MS_CBND', 'MS_CBNDVA', 'MS_d8-THC', 'MS_CBL', 'MS_CBTA-1', 'CBT-1', 'MS_CBTV-1', 'MS_CBTA-3', 'MS_CBT-3', 'MS_CBTV-3', 'MS_CBT-2', 'MS_329-11a', 'MS_329-11b', 'MS_329-11c', 'MS_329-11d', 'MS_329-11e', 'MS_373-12a', 'MS_373-12b', 'MS_373-12c', 'MS_373-12d', 'MS_327-13a', 'MS_327-13b', 'MS_327-13c', 'MS_371-14a', 'MS_371-14b', 'MS_417-15a', 'MS_373-15b', 'MS_373-15c', 'MS_357-16a', 'MS_313-16b', 'MS_361-17a', 'MS_361-17b', 'MS_331-18a', 'MS_331-18b', 'MS_331-18c', 'MS_331-18d', 'MS_375-19a', 'MS_375-19b', 'MS_375-19c']

    return dem_clinic_features, left, right, all_features

def get_feature_importance(data, target_column, features, n_features=15):
    # Process left data
    data = data[features].dropna()
    data[target_column] = data[target_column].apply(transform_column)

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model to your data
    model.fit(X, y)

    # Get feature importance's
    importances = model.feature_importances_

    # Create a dictionary of feature importances with feature names
    importance_dict = dict(zip(X.columns, importances))

    # Sort the features by importance
    sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Extract the names of the top n_features most important features
    top_features = [feature[0] for feature in sorted_importances[:n_features]]
    scores = [feature[1] for feature in sorted_importances[:n_features]]

    return top_features, scores


def tune_randomforest_classifier(data, X_train, X_test, y_train, y_test, f_dict, chemical,
                                 target='pain improvement percentage'):
    # This function tunes a RandomForest classifier using RandomizedSearchCV
    # It also calculates the SHAP values for the best model,
    # and updates the feature importance dictionary if needed (chemical is True)

    # Define the parameter grid to search
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 4, 7],
        'bootstrap': [True, False]
    }
    y_train[target] = y_train[target].apply(transform_column)

    y_test[target] = y_test[target].apply(transform_column)

    data[target] = data[target].apply(transform_column)

    # Create a RandomForest classifier
    rf_model = RandomForestClassifier()

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, scoring='roc_auc', cv=5,
                                       n_iter=10, random_state=42, n_jobs=-1)

    # Fit the model to the training data
    random_search.fit(X_train, np.array(y_train).ravel())

    best_model = random_search.best_estimator_
    best_train_roc_auc = random_search.best_score_
    if chemical:
        explainer = shap.TreeExplainer(best_model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_train)

        sv = shap_values
        aggs = np.abs(sv).mean(1)
        aggs = aggs[0] + aggs[1]

        for col, val in zip(X_train.columns, aggs):
            f_dict[col] = f_dict[col] + np.mean(val)

    # Evaluate the best model on the test set
    y_pred = best_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)

    return auc_score


def plot_top_features(importance_dict, top_n, plot_title):
    # Sort the dictionary based on values in descending order
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Extract the top N features
    top_features = dict(sorted_importance[:top_n])
    for key, value in top_features.items():
        top_features[key] = (int(value * 100)) / 100

    # Create a color gradient for the bars
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))

    # Create a bar plot with customized spacing and styling
    fig, ax = plt.subplots(figsize=(7, 0.6 * top_n))  # Adjust the figure size as needed
    bars = ax.barh(list(top_features.keys()), list(top_features.values()), color=colors, edgecolor='black', height=0.3)

    # Customize the plot appearance
    ax.set_xlabel('Importance')
    ax.set_title(plot_title, fontsize=14, loc='left')
    ax.invert_yaxis()  # Invert the y-axis for descending order

    # Add values to the end of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')

    # Show the plot
    plt.show()

def calculate_differences(scores):
    return scores[:, 1] - scores[:, 0]


def plot_waterfall_and_beeswarm(data, boxplot_values, title_waterfall, title_beeswarm, labels_beeswarm, main_title=None):
    differences = calculate_differences(data)
    sorted_differences = np.sort(differences)[::-1]  # Sort differences from largest to smallest
    sns.set(style="whitegrid")

    # Increase font size
    plt.rcParams.update({'font.size': 12})

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax1 = sns.swarmplot(data=boxplot_values, ax=axes[0])
    ax1.set_title(f'{title_beeswarm}', fontsize=14, loc='left')  # Set title for beeswarm plot and increase font size

    # Adjust the colors of the boxplot
    box_colors = sns.color_palette(n_colors=2)
    ax1 = sns.boxplot(data=boxplot_values, boxprops=dict(facecolor='white', edgecolor='black', linewidth=1), ax=axes[0],
                      width=0.3, palette=box_colors)

    ax1.set_ylabel('AUC scores', fontsize=13)
    ax1.set_xticklabels(labels_beeswarm, fontsize=13)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in
               box_colors]
    labels_with_color = [f'{label}\nMean: {np.mean(boxplot_values[i]):.3f}' for i, label in enumerate(labels_beeswarm)]
    ax1.legend(handles, labels_with_color, loc='upper left')

    # Create a color gradient from red to blue
    colors = plt.cm.RdBu(np.linspace(0, 1, len(sorted_differences)))

    # Plot colored bars from each value to the X axis
    for i, (diff, color) in enumerate(zip(sorted_differences, colors)):
        axes[1].plot([i, i], [0, diff], color=color, linestyle='-', linewidth=4)

    axes[1].set_title(f'{title_waterfall}', fontsize=14, loc='left')
    axes[1].set_xlabel('Index', fontsize=13)
    axes[1].set_ylabel('AUC difference', fontsize=13)

    # Add dashed line for the median of the waterplot with legend
    median_diff = np.median(differences)
    axes[1].axhline(median_diff, color='black', linewidth=1, linestyle='--',
                    label=f'Median difference: {median_diff:.3f}')
    axes[1].axhline(0, color='black', linewidth=1, linestyle='-')  # X axis line
    axes[1].legend(loc='upper right')

    if main_title is not None:
        fig.suptitle(main_title, fontsize=16)

    plt.show()


def plot_beeswarm_for_feature(data, feature, ax, desired_order,desired_colors, target_column='pain improvement percentage'):
    sns.boxplot(x=data[target_column], y=data[feature], order=desired_order, ax=ax, width=0.3, palette=desired_colors)
    ax.set_xticklabels([])

    # Adjust spacing between boxplots
    ax.margins(x=0.1)

    ax.set_title(feature)
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel(f'{feature} Value')

def transform_column_to3(value):
    if value >= 0.20 :
        return 'Pain relief'
    elif value <= 0:
        return 'No pain relief'
    return 'Minor pain relief'

