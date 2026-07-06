#Compte rendu des meetings
## Meeting du 26/05/26
** Fait :**
- mise en place et administratif
- lecture et test code alphacsc
- lecture de papiers           
** À Faire : **
- Obtenir un signal y appliquer des fonctions de alphacsc
- adapter le code `python/main.py` du dépôt `NMF_CDL` au cas monocanal            
** Questions :**
- À quoi correspond le paramètre n_times_atom
- Pourquoi un dictionnaire parcimonieux permet d'isoler les motifs
- Pourquoi dans l'exemple filtrer les basses fréquences uniquement
- Comment avoir un signal ECG monocanal
## Meeting du 8/06/26
** Fait :**      
- Extraction de signaux ECG   
- Essais de détermination d'atomes avec différents paramètres  
** À Faire :**     
- Explorer le dépot de jean babtiste
- Continuer les essais et pousser plus en avant les comparaisons
** Questions :**      
- Quel est l'intérêt du paramètre random_state
- Comment augmenter la vitesse de traitement
## Meeting du 23/06/26 :                           
** Fait :**         
- application de plusieurs classifieurs à différents jeux de données                        
- optimisation des taux d'erreur de classification                        
- lecture : https://fr.mathworks.com/help/wavelet/ug/ecg-classification-using-wavelet-features.html#XpwWaveletMLExample-11                    
            https://www.researchgate.net/publication/4234035_ECG_Feature_Extraction_and_Classification_Using_Wavelet_Transform_and_Support_Vector_Machines               
** À Faire ** :                                        
- adapter le code du 1er site pour extraire les features principales et trier par rapport à celles-ci                             
- voir les autres sites                                        
** Questions : **                          
- Y a t-il des possiblillités simple de passer de matlab à python
## Meeting du 29/06/26 :                     
** Fait :**                                            
- Correction d'erreurs précédentes                                  
- test du processus d'extractions de features à partir de la transformée en ondelettes                    
- reconstruction des signaux à partir des atomes                         
** À Faire :**                               
- Passer en matlab plutôt qu'en python pour les ondelettes                              
- Regarder le traitement par réseau de neurones sur matlab directement                      
