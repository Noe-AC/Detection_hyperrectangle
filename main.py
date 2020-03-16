import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import timeit # Pour avoir la durée de calcul d'un calcul
import scipy.stats as st
from scipy.special import comb

np.set_printoptions(linewidth=np.inf)

####################################################
####################################################

		# Définition de diverses fonctions

def factoriel(n): # n!
	return np.math.factorial(n)
def choose_n_k(n,k): # C(n,k) = n!/((n-k)!k!)
	return comb(N=n, k=k, exact=False)
def loi_binomiale(n,x,p): # P(X=x)
	return st.binom.pmf(k=x,n=n,p=p)
def mu_loi_binomiale(n,p): # espérance de la loi binomiale
	mu = n*p
	return mu
def var_loi_binomiale(n,p): # variance de la loi binomiale
	q = 1-p
	sigma2 = n*p*q
	return sigma2
def sigma_loi_binomiale(n,p):
	return math.sqrt(var_loi_binomiale(n=n,p=p))
def cote_Z(x,mu,sigma):
	return (x-mu)/sigma
# Une fonction qui donne un array population [0,1,2,...,n] et l'autre les probabilités [P(X=0),P(X=1),...,P(X=n)] qu'il y ait exactement x bons pixels dans une image
def population_and_weights_binomial_law(n,p):
	population = list(range(n+1))
	weights = []
	for x in population:
		prob = loi_binomiale(n=n,x=x,p=p)
		weights.append(prob)
	return population,weights
def loi_normale_densite(x,mu,sigma):
	return exp(-(x-mu)*(x-mu) / (2*sigma*sigma)) / (sigma * math.sqrt(2*math.pi))
def loi_normale_prob_Z_moins_que_z(z): # P(Z<z)
    return ( 1 + math.erf(z/math.sqrt(2)) )/2
def loi_normale_prob_Z_entre_a_et_b(a,b): # P(a<Z<b)
	return loi_normale_prob_Z_moins_que_z(b) - loi_normale_prob_Z_moins_que_z(a)
def loi_normale_prob_Z_plus_que_z(z) : # P(Z>z)
	return 1-loi_normale_prob_Z_moins_que_z(z)
def cote_Z(x,mu,sigma):
	return (x-mu)/sigma
def loi_normale_prob_X_moins_que_x(x,mu,sigma):
	return loi_normale_prob_Z_moins_que_z(cote_Z(x,mu,sigma))
def loi_normale_prob_X_entre_a_et_b(a,b,mu,sigma):
	return loi_normale_prob_X_moins_que_x(b,mu,sigma) - loi_normale_prob_X_moins_que_x(a,mu,sigma)
def loi_normale_prob_X_plus_que_x(x,mu,sigma):
	return 1-loi_normale_prob_X_moins_que_x(x,mu,sigma)
def loi_de_Poisson(x,l): # pour x=0,1,2,3,...
	return math.exp(-l)*math.pow(l,x)/factoriel(x)
def chi2(a):
	a = np.array(a) # taille (m,n)
	m,n = a.shape # m est la longueur de la variable X, n est la longueur de la variable Y
	somme = np.sum(a) # = sum(Y_sum)
	X_sum = np.sum(a,axis=0) # vecteur de longueur n
	Y_sum = np.sum(a,axis=1) # vecteur de longueur m
	X_sum = np.reshape(X_sum,newshape=(1,n)) # taille (1,n)
	Y_sum = np.reshape(Y_sum,newshape=(m,1)) # taille (m,1)
	a_theorique = X_sum*Y_sum/somme # taille (m,n), la table théorique
	nu = (m-1)*(n-1) # nombre de degrés de libertés
	chi2_somme = np.sum((a_theorique-a)**2/a_theorique)
	return chi2_somme, nu

####################################################
####################################################

		# Terminologie

"""
pct  = "probabilité critique totale". C'est la probabilité de détecter quelque chose dans du bruit sur l'ensemble des grilles. Typiquement, pct=0.05
pcg  = "probabilité critique grille". C'est la probabilité de détecter quelque chose dans du bruit sur une grille donnée
pcc  = "probabilité critique case".   C'est la probabilité de détecter quelque chose dans du bruit dans une case donnée.
ncm  = "nombre de cases maximales".   C'est le nombre de cases de la grille ayant la plus grande résolution
nce  = "nombre de cases éligibles".   C'est le nombre de cases contenant assez de points pour pouvoir atteindre Z_crit
ncnv = "nombre de cases non vides".   C'est le nombre de cases non vides sur une grille donnée.
"""

####################################################
####################################################

		# Initialisation des données

# On considère m points en dimension n à valeurs binaires {0,1}

# Les paramètres avec lesquels je peux jouer :

n = 2    # n= nombre de dimensions continues
m = 400 # m = nombre de points
pct = 0.05 # probabilité que sur l'ensemble de toutes les grilles on détecte quelque chose dans du bruit

nombre_de_tests = 100

def m_to_res(m): # de sorte qu'il y ait au minimum 28 points par boîte
	res_max = 20
	res_min = int(np.floor(np.sqrt(m/27.71)))
	return np.min([res_min,res_max])
resolution_max = m_to_res(m)
#resolution_max = 10 # dans chaque dimensions, on coupe au plus resolution_max fois
# Pour l'instant ici ça ne marche que avec n=2
ncm = resolution_max**n # ncm = nombre de cases maximales
resolutions = np.ones(shape=n,dtype=int) # taille n, la résolution des n variables continues

print("n =",n)
print("m =",m)
print("pct =",pct)
print("Resolution =",resolution_max)

# On se donne des données
X = np.zeros((m,n)) # position des m points selon les n variables continues
y = np.zeros(m,int) # valeurs binaires {0,1} des m points

# On initialise les points avec des valeurs aléatoires (optionnel) :
p=0.5 # probabilité qu'un point soit un 1
q=1-p # probabilité qu'un point soit un 0


count_case_exceed = 0


# On se donne une distribution uniforme de points dans un carré [0,1] :
X = np.random.random((m,n)) # taille (m,n) de nombres aléatoires en [0.0, 1.0[



smiley_or_not = 0
if smiley_or_not==1:
	y = np.random.choice(a=[0,1], p=[q,p], size=m) # taille m de nombres aléatoires en {0,1}
	shift_eye_hor = 0.15
	shift_eye_ver = 0.15
	for i in range(m):
		u,v = tuple(X[i]) # coord. (u,v) en [0,1]x[0,1] en R^2
		if 0.45**2<=(u-0.5)**2+(v-0.5)**2 and (u-0.5)**2+(v-0.5)**2<=0.5**2:
			y[i] = 0
		if 0.25**2<=(u-0.5)**2+(v-0.5)**2 and (u-0.5)**2+(v-0.5)**2<=0.35**2 and u<0.4:
			y[i] = 0
		if (u-0.5-shift_eye_ver)**2+(v-0.5+shift_eye_hor)**2<=0.10**2: # oeil gauche
			y[i] = 0
		if (u-0.5-shift_eye_ver)**2+(v-0.5-shift_eye_hor)**2<=0.10**2: # oeil droit
			y[i] = 0




# On regarde le min et le max dans chaque dimensions
minimums = np.min(X,axis=0) # taille n, c'est le min dans chaque variables continues
maximums = np.max(X,axis=0) # taille n, c'est le max dans chaque variables continues


number_detected = []
for test_number in range(nombre_de_tests):

	if smiley_or_not==0:
		y = np.random.choice(a=[0,1], p=[q,p], size=m) # taille m de nombres aléatoires en {0,1}

	# Calcul de la proportion du nombre de 1
	nombre_de_1 = np.count_nonzero(y)
	nombre_de_0 = m - nombre_de_1
	proportion = nombre_de_1 / m # proportion de points qui sont des 1

	# On se donne un compteur pour le nombre de fois où on détecte quelque chose d'anormal
	count_case_detected = 0


	if smiley_or_not==0:
		range_resolution = range(1,resolution_max+1)
	if smiley_or_not==1:
		range_resolution = range(resolution_max,resolution_max+1) # on ne prend que la résolution maximale


	for resolution_0 in range_resolution:
		for resolution_1 in range_resolution:
			if resolution_0*resolution_1>1:
				resolutions[0] = resolution_0
				resolutions[1] = resolution_1
				nb_cases = np.prod(resolutions)

				# On découpe les variables continues en intervalles discrets
				pas = (maximums-minimums)/resolutions # On partitionne les n variables continues en intervalles de longueur maximums-minimums/resolutions

				# On change les variables continues en variables discrètes (i_1,...,i_n) entières disant dans quel interval le point se trouve
				X_cat = np.zeros((m,n),int) # cat pour catégorie
				bins = [] # c'est une liste qui contient les bin_j pour chaque dimension
				for j in range(n):
					min_j = minimums[j]
					max_j = maximums[j]
					res_j = resolutions[j]
					bin_j = np.linspace(start=min_j,stop=max_j,num=res_j+1)
					bins.append(bin_j)
				for j in range(n):
					X_j   = X[:,j] 
					bin_j = bins[j]
					ind_j  = np.digitize(X_j,bin_j) # min <= x < max
					res_j = resolutions[j]
					ind_j[ind_j==res_j+1] = res_j # pour x==max on doit corriger la catégorie
					X_cat[:,j] = ind_j-1 # on commence les indices à 0 et non à 1

				# On dénombre il y a combien de points dans chaque catégories
				# On crée un array sparse de 0 pour chaque catégorie
				grid_tot = np.zeros(resolutions,int) # taille (resolutions) = (resolutions[0],resolutions[1],...,resolutions[n-1]), c'est le nombre de points total dans chaque case
				grid_1   = np.zeros(resolutions,int) # taille (resolutions), on y met le nombre de valeurs 1 dans la case
				for i in range(m):
					coord_cat = tuple(X_cat[i,:]) # coordonnées catégoriques (i_1,...,i_n) du point
					grid_tot[coord_cat] += 1 # on augmente le nombre de points
					if y[i]==1:grid_1[coord_cat] += 1 # on augmente le nombre de points à valeurs 1

				# On regarde quelles cases sont non vides
				cases_non_vides = np.nonzero(grid_tot)
				ncnv = len(cases_non_vides[0]) # ncnv = nombre de cases non vides. Ce nombre est forcément <= m car c'est le nombre de cases non vides

				# On calcule le nombre de points dans les cases
				m_max = np.max(grid_tot) # nombre de points dans la case ayant le plus grand nombre de points
				m_min = np.min(grid_tot) # nombre de points dans la case ayant le moins de points
				m_moy = m/ncnv

				# On se donne un critère pour savoir s'il se passe quelque chose ou non dans une case
				pcg = 1-(1-pct)**(1/(ncm-1)) # probabilité critique pour une grille
				pcc = 1-(1-pcg)**(1/ncnv) # probabilité critique pour une case
				chi2_crit = st.chi2.isf(pcc,1)
				Z_crit    = -st.norm.ppf(pcc/2) # Z_crit est le Z critique où il se passe quelque chose dans une case
				m_crit    = int(np.ceil(Z_crit**2)) # nombre de points minimum critique nécessaire dans une case pour qu'il se passe quelque chose

				cases_eligibles = np.transpose(np.nonzero(grid_tot>=m_crit)) # les cases éligibles, i.e. ayant >= m_crit points
				nce = len(cases_eligibles) # nce : nombre de cases éligibles, i.e. ayant un nombre de points >= m_crit

				# On calcule la cote Z là où la case est éligible
				grid_Z        = np.zeros(resolutions)      # taille (resolutions), on y met la cote Z selon la loi binomiale
				grid_Z_exceed = np.zeros(resolutions,bool) # taille (resolutions), on y met 1 si |Z|>Z_crit, ou 0 sinon


				for i in range(nce):
					coord_case_eligible = tuple(cases_eligibles[i])
					m_pos = grid_tot[coord_case_eligible]
					x_pos = grid_1[coord_case_eligible]

					dedans_1 = x_pos
					dedans_0 = m_pos - dedans_1
					dehors_1 = nombre_de_1 - dedans_1
					dehors_0 = nombre_de_0 - dedans_0
					tableau_de_contingence_case = [[dedans_1,dedans_0],[dehors_1,dehors_0]]
					chi2_case_observe,nu_case   = chi2(a=tableau_de_contingence_case)
					
					mu_pos    = mu_loi_binomiale(n=m_pos,p=proportion)
					sigma_pos = sigma_loi_binomiale(n=m_pos,p=proportion)
					Z_top     = (x_pos-mu_pos-1/2)/sigma_pos # le 1/2 est pour le passage binomial -> normal
					Z_low     = (x_pos-mu_pos+1/2)/sigma_pos # le 1/2 est pour le passage binomial -> normal
					Z         = (Z_top+Z_low)/2

					print_details_or_not = 0
					if print_details_or_not==1:
						print("")
						print("n_pos",m_pos)
						print("proportion",proportion)
						print("mu_pos",mu_pos)
						print("sigma_pos",sigma_pos)
						print("nce",nce)
						print("pcg",pcg)
						print("pcc",pcc)
						print("x_pos",x_pos)
						print("Z_low",Z_low)
						print("Z_top",Z_top)
				
					if chi2_case_observe>=chi2_crit:
						grid_Z_exceed[coord_case_eligible] = True
						count_case_exceed += 1
					grid_Z[coord_case_eligible] = Z

				# On regarde s'il existe une case qui sort de l'ordinaire
				nb_cases_exceed = np.count_nonzero(grid_Z_exceed)
				Z_max           = np.max(grid_Z)
				Z_min           = np.min(grid_Z)
				Z_abs           = np.abs(grid_Z)
				Z_abs_max       = np.max(Z_abs)
				


				if nb_cases_exceed>0:
					count_case_detected += nb_cases_exceed

				print_or_not = 0
				if nb_cases_exceed>0 or print_or_not:
					print("\nTest ",test_number," sur ",nombre_de_tests," tests")
					print("Résolutions =",resolutions)
					print("p =",p)
					print("Proportion de 1 sur tous les points =",proportion)
					print("Z_abs_max =",Z_abs_max)
					print("nb_exceed = ",nb_cases_exceed)
					print("pct =",pct)
					print("pcg =",pcg)
					print("pcc =",pcc)
					print("Nombre de pts. maximum dans une case =",m_max)
					print("Nombre de pts. minimum dans une case =",m_min)
					print("Nombre de pts. moyen dans une case =",m_moy)
					print("m_crit =",m_crit)
					print("ncm : Nombre de cases max =",ncm)
					print("nce : Nombre de cases éligibles =",nce)
					print("ncnv : Nombre de cases non vides =",ncnv)
					print("Nombre de cases =",nb_cases,"\n")
				
				plot_or_not = 1
				if nb_cases_exceed>0 and plot_or_not==1:
					minimums_x = bins[0][0:-1]
					minimums_y = bins[1][0:-1]
					maximums_x = bins[0][1:]
					maximums_y = bins[1][1:]
					mids_x = (maximums_x+minimums_x)/2
					mids_y = (maximums_y+minimums_y)/2
					fig, ax = plt.subplots()
					x_min = minimums_x[0]
					x_max = maximums_x[-1]
					y_min = minimums_y[0]
					y_max = maximums_y[-1]
					extent=[x_min,x_max,y_max,y_min]
					# Horizontalement c'est la colonne 1, verticalement c'est la colonne 0
					cMap = ListedColormap(['blue', 'red']) # 0:bleu, 1:rouge
					if smiley_or_not==0:
						plt.scatter(x=X[:,1],y=X[:,0],c=y,s=0.5,cmap=cMap,alpha = 100*ncnv/m)
					if smiley_or_not==1:
						plt.scatter(x=X[:,1],y=X[:,0],c=y,s=0.5,cmap=cMap,alpha = 3000*ncnv/m)
					heatmap = ax.imshow(X=np.abs(grid_Z),cmap='summer',extent=extent,alpha=1,vmin=0,vmax=np.max(np.abs(grid_Z))) # pour afficher la valeur absolue de Z
					cbar = plt.colorbar(heatmap)
					cbar.ax.set_ylabel('valeur absolue de la cote Z', rotation=90)
					for i in range(len(mids_x)):
					    for j in range(len(mids_y)):
					    	Z = grid_Z[i,j]
					    	Z_text = "%.2f"%Z
					    	text = ax.text((j+1/2)/len(mids_y),(i+1/2)/len(mids_x),Z_text,ha="center", va="center", color="black")
					ax.set_title("Cote Z de chaque catégorie")
					plt.xlabel("x : colonne 1 de l'array X_cat")
					plt.ylabel("y : colonne 0 de l'array X_cat")
					plt.xlim(x_min,x_max)
					plt.ylim(y_min,y_max)
					plt.title("Catégories de X_cat et cotes Z")
					plt.show()

	print("Test =",test_number,"\tNombre de régions détectées =",count_case_detected)
	number_detected.append(count_case_detected)

number_detected_sum = sum(number_detected)
number_detected_len = len(number_detected)
number_detected_moy = number_detected_sum / number_detected_len
print("Nombre de détections moyennes =", number_detected_moy)
print("count_case_exceed/nombre_de_tests",count_case_exceed/nombre_de_tests)








