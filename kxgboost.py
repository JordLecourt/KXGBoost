import os
import xgboost
import pickle
import pandas as pd
from pykrige.compat import Krige, check_sklearn_model
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import RegressorChain

class KXGBoost:

    def __init__(
        self,
        tree_method='auto',
        gpu_id=-1,
        regression_model=None,
        multi_output=False,
        n_output=1,
        output_order=None,
        method="ordinary",
        variogram_model="linear",
        n_closest_points=10,
        verbose=False,
        coordinates_type="euclidean"
    ):
        """
        Construis un objet de type KXGBoost
        
        :param tree_method: defaut = 'auto'. Si un gpu est utilise, selectionner 'gpu_hist'
        :param gpu_id: defaut = -1. Si -1, il n'y a pas de gpu.
        :param regression_model: defaut = None. Pour utiliser un autre modele que XGBoost
        :param multi_ouput: defaut = False. y a-t-il plusqu'une caracteristique?
        :param n_output: defaut = 1. Si multi_output = True, combien de caracteristiques y a-t-il?
        :param method: defaut='ordinary. Methode utilise pour le krigeage
        :param variogram_model: defaut = 'linear'. Variogramme utilise pour le krigeage
        :param output_order: defaut = None. L'ordre dans lequel on predit les caracteristiques
        :param n_closest_points: defaut = 10. Nombre de points les plus proches a considerer
        :param coordinates_type: defaut = 'euclidean'. Soit 'euclidean' (x, y) ou 'geographic' (lat, lng)
        """
        self.verbose_int = int(verbose)
        self.verbose = bool(verbose)

        if regression_model is None:
            self.reg = xgboost.XGBRegressor(
                    tree_method=tree_method, 
                    gpu_id=gpu_id
                )
        else:
            check_sklearn_model(regression_model)
            self.reg = regression_model

        self.multi_output = multi_output
        self.n_output = n_output

        if (multi_output):
            self.pipeline = RegressorChain(base_estimator=self.reg, order=output_order)

            self.krige = []
            for _ in range(n_output):
                self.krige.append(Krige(
                    method=method,
                    variogram_model=variogram_model,
                    n_closest_points=n_closest_points,
                    verbose=self.verbose,
                    coordinates_type = coordinates_type
                ))
        else:
            self.krige = Krige(
                            method=method,
                            variogram_model=variogram_model,
                            n_closest_points=n_closest_points,
                            verbose=self.verbose,
                            coordinates_type=coordinates_type
                        )

    def set_params(self, params):
        """
        Permet de changer les paramètres du regresseur
         
        :param params: dictionnaire des parametres a utiliser
        """
        if (self.multi_output):
            self.pipeline.set_params(**params)
        else:
            self.reg.set_params(**params)

    def fit(self, x, coords, y):
        """
        Entraine le modele sur les donnees passees en parametres.

        :param x: dataframe des donnees utilisees par l'algorithme a base d'arbres. doit inclure
           latitude et longitude, mais il peut y avoir d'autres caracteristiques egalement
        :param coords: dataframe des donnees utilisees par le krigeage. doit etre uniquement les
                coordonnees. (x, y) si coordinates_type = 'euclidean' ou (lng, lat) si
                coordinates_type = 'geographic'
        :param y: dataframe des caracteristiques.
        """
        if isinstance(coords, pd.DataFrame): coords = coords.values
        if isinstance(x, pd.DataFrame): x = x.values
        if isinstance(y, pd.DataFrame): y = y.values

        if(self.multi_output):
            self.pipeline.fit(x, y)
            y_pred = self.pipeline.predict(x)
            if (self.verbose) : print('Finished learning regression model')

            for i in range(self.n_output):
                self.krige[i].fit(x=coords, y=y[:,i] - y_pred[:,i])
                if (self.verbose) : print('Finished kriging residuals ', i)

        else:
            self.reg.fit(x, y)
            y_pred = self.reg.predict(x)
            if (self.verbose) : print('Finished learning regression model')

            self.krige.fit(x=coords, y=y - y_pred)
            if (self.verbose) : print('Finished kriging residuals ')

    def predict(self, x, coords):
        """
        Predit les variables dependante avec un modele entraine

        :param x: dataframe des donnees utilisees par l'algorithme a base d'arbres. Doit inclure
           latitude et longitude, mais il peut y avoir d'autres caracteristiques egalement
        :param coords: dataframe des donnees utilisees par le krigeage. doit etre uniquement les
                coordonnees. (x, y) si coordinates_type = 'euclidean' ou (lng, lat) si
                coordinates_type = 'geographic'
        """
        if isinstance(coords, pd.DataFrame): coords = coords.values
        if isinstance(x, pd.DataFrame): x = x.values
        
        if (self.multi_output):
            y_pred = self.pipeline.predict(x)
            for i in range(self.n_output):
                y_pred[:, i] += self.krige[i].predict(coords)

        else:
            y_pred = self.reg.predict(x) + self.krige.predict(coords)

        return y_pred

    def grid_search_tune_parameters(
        self,
        x,
        coords,
        y,
        param_grid,
        n_iter=25,
        cv=5,
        verbose=1,
        scoring='l2',
        n_jobs=1,
        refit=True
    ):
        """
        Permets de tester plusieurs combinaisons de parametres afin de trouver la meilleure

        :param x: dataframe des donnees utilisees par l'algorithme a base d'arbres. Doit inclure
           latitude et longitude, mais il peut y avoir d'autres caracteristiques egalement
        :param coords: dataframe des donnees utilisees par le krigeage. doit etre uniquement les
                coordonnees. (x, y) si coordinates_type = 'euclidean' ou (lng, lat) si
                coordinates_type = 'geographic'
        :param y: dataframe des caracteristiques.
        :param param_grid: dictionnaire de parametres a tester
        :param n_iter: defaut = 25. Nombre de combinaison aleatoire a tester
        :param cv: defaut = 5. Cross validation. Nombre de groupe sur lequel tester la performance
        :param scoring: defaut = 'l2'. Technique utiliser pour evaluer les modeles
        :param n_jobs: defaut = 1. Nombre d'essais executes en parallele. Sert a augmenter la rapidite
                des tests. n_jobs = -1 permet d'avoir le nombre maximum
        :param refit: defaut = True. Doit-on reentrainer le modele avec les parametres trouves. Si
               scoring est une liste, refit doit etre la metrique a utiliser pour le reentrainement
        """
        if (self.multi_output):
            clf = RandomizedSearchCV(self.pipeline, 
                        param_distributions = param_grid, 
                        n_iter = n_iter, 
                        scoring = scoring, 
                        n_jobs = n_jobs,
                        verbose = verbose,
                        cv=cv,
                        refit=refit
                    )
            clf.fit(x, y)

            if refit:
                self.pipeline = clf.best_estimator_
                self.fit(x, coords, y)
            
        else:
            clf = RandomizedSearchCV(self.reg, 
                        param_distributions = param_grid, 
                        n_iter = n_iter, 
                        scoring = scoring, 
                        n_jobs = n_jobs,
                        verbose = verbose,
                        cv=cv,
                        refit=refit
                    )
            clf.fit(x, y)

            if refit:
                self.reg = clf.best_estimator_
                self.fit(x, coords, y)

        return clf

    def save(self, path):
        """
        Permets de sauvegarder un modele entraine afin de le reutiliser ulterieurement.
        Puisque le modele est consitue d'au moins deux fichiers, path doit être
        le nom d'un dossier plutot qu'un fichier
                
        :param path: le chemin du dossier dans lequel on sauvegarde le modele. Le modele est
              constitue de plusieurs fichiers, donc il est conseille de fournir un dossier
              vierge (ou inexistant) et bien identifie (ex model1)
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
        if self.multi_output :
            pickle.dump(self.pipeline, open(path + '/reg_model.sav', 'wb'))
            for i in range(self.n_output):
                pickle.dump(self.krige[i], open(path + '/krige_model_'+ str(i) +'.sav', 'wb'))

        else:
            pickle.dump(self.reg, open(path + '/reg_model.sav', 'wb'))
            pickle.dump(self.krige, open(path + '/krige_model.sav', 'wb'))

    def load(self, 
            path, 
            multi_output=False, 
            n_output=1
    ):
        """
        Permet de charger un modele

        :param path: le chemin du dossier dans lequel le modele est sauvegarde
        :param multi_ouput: defaut = False. y a-t-il plusqu'une caracteristique?
        :param n_output: defaut = 1. Si multi_output = True, combien de caracteristiques y a-t-il?
        """
        self.multi_output = multi_output
        self.n_output = n_output

        if multi_output :
            self.pipeline = pickle.load(open(path + '/reg_model.sav', 'rb'))
            for estimator in self.pipeline.estimators_ :
                estimator.enable_categorical = False
            self.krige = []
            for i in range(n_output):
                self.krige.append(pickle.load(open(path + '/krige_model_'+ str(i) +'.sav', 'rb')))

        else:
            self.reg = pickle.load(open(path + '/reg_model.sav', 'rb'))
            self.krige = pickle.load(open(path + '/krige_model.sav', 'rb'))