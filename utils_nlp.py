from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
from sklearn.mixture import BayesianGaussianMixture
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import re, functools, logging, re
import pandas as pd
import numpy as np
from pandas.core.base import PandasObject
import pdb       # debugger
from scipy.stats import genextreme as gev
from gensim.parsing.preprocessing import strip_short
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from copy import deepcopy
#import spacy, en_core_web_sm


# Mimics R's paste() function for two lists:
#---------------------------------------------
def reduce_concat(x, sep=""):
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)

def paste(*lists, sep=" ", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)
#----------------------------------------------------


def jitter(M: int, noise_scale: float = 10**5., seed : int = None):
    
  """ Generates jitter that can be added to any float, e.g.
      helps when used with pd.qcut to produce unique class edges
      M: number of random draws, i.e. size
  """  
  if seed is not None:
     np.random.seed(seed)
  return np.random.random(M)/noise_scale ;


def switch(key, switcher = {'years': 'Y', 'days': 'D', 'weeks': 'W','months': "M"}):
        return switcher.get(key, "Key does not match any in 'switcher' argument.")  


class preprocess_text(BaseEstimator, TransformerMixin):    
   """
   This is an extension of the clean_text class below. 
   Stemming and stopwords filtering were added for preprocessing.
   """ 
   def __init__(self, input_col, output_col, stemming = True, unique_tokens = True):
      self.name = output_col
      self.input_col = input_col
      self.stemming = stemming
      self.unique_tokens = unique_tokens
      if self.unique_tokens : print("Deduplicating tokens per document/sentence.")
      self.regexp = re.compile('(?u)(?:(?!\d)\w)+\\w+')        # remove punctuation and digits
      if self.stemming : self.stemmer = PorterStemmer() ; print("Applying Porter stemming.")
      self.stop_words = set(stopwords.words('english'))

   def fit(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      return self

   def transform(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      self.df_raw = deepcopy(X)
      d = self.df_raw[self.input_col].tolist()
      for i in range(len(d)):
        d[i] = " ".join(self.regexp.findall(str(d[i]))).lower().strip()        # strip ws and convert to lower
        tokens = word_tokenize(d[i])
        if self.unique_tokens:
            # Make tokens unique in sentence
            #tokens = list(set(tokens))   # this loses word order
            my_unique_sentence = []
            e = [my_unique_sentence.append(tok) for tok in tokens if tok not in my_unique_sentence]
            tokens = my_unique_sentence
 
        # Stem words and remove stopwords:
        if self.stemming:
           filtered_tokens = [self.stemmer.stem(w) for w in tokens if not w in self.stop_words]
        else:
           filtered_tokens = [w for w in tokens if not w in self.stop_words]
        d[i] = " ".join(filtered_tokens)
      self.df_raw[self.name] = d 
      return self.df_raw


class clean_text(BaseEstimator, TransformerMixin):
    
   def __init__(self, input_col, output_col):
      self.name = output_col
      self.input_col = input_col
      self.regexp = re.compile('(?u)(?:(?!\d)\w)+\\w+')        # remove punctuation and digits

   def fit(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      return self

   def transform(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      self.df_raw = deepcopy(X)
      d = self.df_raw[self.input_col].tolist()
      for i in range(len(d)):
        d[i] = " ".join(self.regexp.findall(str(d[i]))).lower().strip()        # strip ws and convert to lower
      self.df_raw[self.name] = d 
      return self.df_raw
    
#-------------------------------------------------------------------------------------------------------------------    

class StringCleaner(TransformerMixin):

    def __init__(self, mapping = {},  replace = True, verbose = True):
        # example mapping ={'column_name':{'find':'replacement}}
        self.replace = replace
        #self.convert_cols = convert_cols
        self.mapping = mapping
        self.verbose = verbose
    def fit(self):
        pass
    
    def transform(self, X, minimum_length = 5):
         if isinstance(X, pd.Series):
            return X
         if isinstance(X, pd.DataFrame):
            X_temp = deepcopy(X)
            for col in self.mapping:
                X_temp[col] = X_temp[col].str.lower()
              
            for col in self.mapping:
                for word in self.mapping[col]:
                    X_temp[col] = X_temp[col].str.replace(word,self.mapping[col][word])
            if self.verbose:
                print("Replaced strings in columns: "+ str([col for col in self.mapping]))
            return X_temp
            
"""
def doc_tagger(sentence): 
    
    """'Tag document with linguistic annotations'"""
    
    nlp = en_core_web_sm.load()
    doc = nlp(str(sentence))
    token_text, token_lemma, token_pos, token_tag, token_dep, nounIndices = [], [], [], [], [], []
    index, i = 2*[0]
    annotations = {}
    
    # Linguistic annotations:
    for token in doc:
        token_text.append(token.text) ; token_lemma.append(token.lemma_)
        token_pos.append(token.pos_) ; token_tag.append(token.tag_)
        token_dep.append(token.dep_) ; 
        
        annotations.update({i : [token.lemma_, token.pos_, token.dep_, token.shape_]})
        i += 1
    
    token_char = {'orig token' : token_text, 'pos' : token_pos, 'nounIndices' :nounIndices, 
                  'tag' : token_tag, 'dep': token_dep,  'lemma token' : token_lemma}  
    
    return annotations #, token_char 
"""

#-------------------------------------------------------------------------------------------------------------------    

def get_cases(descr, search_list_noun = ['truck', "car", 'vehicle', 'third party',"third", \
                                         'insured', 'van', 'driver', 'traffic', 'rear'], \
                     search_list_verb = ['drove', 'collided', 'struck', 'hit','parked', "reversed", 'turned'], \
              thresh = 1): 
   """
   Input:
   descr: claims descr. column
   search_list_noun:    search list with nouns to search for in the claims descr
   search_list_verb:    search list with verbs to search for 
   thresh: threshold parameter; overall number of hits, in both nouns and verbs search list 
   """ 
        
   def is_phrase_in(phrase, text):
    """Exact string matcher"""
    return re.search(r"\b{}\b".format(phrase), text, re.IGNORECASE) is not None ;

   weights = {'noun_weight' : 0, 'vrb_weight' : 0}

   for i in search_list_noun:
        if is_phrase_in(i, descr):
          weights['noun_weight'] +=1  
   for i in search_list_verb:
        if is_phrase_in(i, descr):
          weights['vrb_weight'] +=1  
        
   return (weights['noun_weight'] + weights['vrb_weight']) > thresh

    
#----------------------------------------------------------------------------------------------------------------------    

def create_logger(**kwargs):
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
 
    # create the logging file handler
    fh = logging.FileHandler(**kwargs)
 
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
 
    # add handler to logger object
    logger.addHandler(fh)
    return logger

# Set global settings in Jupyter notebook for convenience
#---------------------------------------------------------------------------------------------------------
def start():
    options = {
        'display': {
            'max_columns': 500,
            #'max_colwidth': 25,
            #'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 500,
            #'max_seq_items': 50,         # Max length of printed sequence
            'precision': 4,
            'width' : 1000
            #'show_dimensions': False
        },
        'mode': {
            'chained_assignment': None   # Controls SettingWithCopyWarning
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+

#---------------------------------------------------------------------------------------------------------------

def make_colnames(df, inplace = False):
    """Reformat column names for compatibility with SQl server"""
    regexp = re.compile('(?u)(?:(?!\d)\w)+\\w+')        # remove punctuation and digits
    d = df.columns
    cols = [];
    for i in range(len(d)):
        a = "_".join(regexp.findall(str(d[i]))).lower().strip()       # strip ws and convert to lower; set underscores
        cols.append(a)
    df.columns = cols
    if inplace is not True:
      return df
PandasObject.make_colnames = make_colnames   

#---------------------------------------------------------------------------------------------------------------

class train_test_split_extend(BaseEstimator, TransformerMixin):

  """ Partition the data into training/validation/testing subsets"""

  def __init__(self):
      return None;

  def fit(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      return self

  def transform(self, X, y=None, sizes=[], random_state=None, shuffle=True):

    assert isinstance(sizes, list)
    assert isinstance(X, pd.DataFrame)
    self.X = X
    self.y = y
    self.sizes = sizes
    self.random_state = random_state
    self.shuffle = shuffle

    if y is None: 

        if len(self.sizes) == 3:
          p_rest = round(1-self.sizes[0],2) ; p_test = self.sizes[2]/p_rest
          X_train, X_rest = train_test_split(self.X, test_size = p_rest, random_state = self.random_state, shuffle = self.shuffle)
          X_valid, X_test = train_test_split(X_rest, test_size=p_test, random_state=self.random_state,shuffle=self.shuffle)
          assert (self.X.shape[0] == X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
          return X_train, X_valid, X_test

        if len(self.sizes) == 2:
          p_test = round(1-self.sizes[0],2) ;
          X_train, X_test = train_test_split(self.X, test_size = p_test, random_state=self.random_state,shuffle=self.shuffle)
          assert (self.X.shape[0] == X_train.shape[0] + X_test.shape[0])
          return X_train, X_test

    else:    
        assert isinstance(y, pd.DataFrame)
        if len(self.sizes) == 3:
          p_rest = round(1-self.sizes[0],2) ; p_test = self.sizes[2]/p_rest
          X_train, X_rest, y_train, y_rest = train_test_split(self.X, self.y, stratify = self.y, test_size = p_rest, random_state = self.random_state, shuffle = self.shuffle)
          X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, stratify = y_rest, test_size=p_test, random_state=self.random_state,shuffle=self.shuffle)
          assert (self.X.shape[0] == X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
          return X_train, X_valid, X_test, y_train, y_valid, y_test;

        if len(self.sizes) == 2:
          p_test = round(1-self.sizes[0],2) ;
          X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify = self.y, test_size = p_test, random_state=self.random_state,shuffle=self.shuffle)
          assert (self.X.shape[0] == X_train.shape[0] + X_test.shape[0])
          return X_train, X_test, y_train, y_test;
        

# Calculate and print lift graph:
#-------------------------------------------------------------------------------------------------------------------------------
def calc_lift(y_prob: np.ndarray, y_actual: pd.DataFrame, bins: int = 100, bins2: int = 5, plot_lift = False, duplicates = "raise", title = None):

    """Input:
       y_prob: posterior probs for y=1
       y_actual: true labels
       
       Return:  
       lift_df: True and predicted subro rate, lift value per bin and overall avg. subro rate as baseline
       lift_index_20: Lift indexes for quintiles (20% bins)
       title:   title for plotly lift chart
    """
    y_actual.index = range(len(y_actual))
    cols = ['ACTUAL','PROB_POSITIVE'] 
    data = [y_actual,y_prob] 
    df = pd.DataFrame(dict(zip(cols,data)))
    total_positive_n = df['ACTUAL'].sum()                     
    total_n = len(df.index)                                    #Total Observations
    natural_positive_prob = total_positive_n/float(total_n)         # overall subro rate

    # Normalize estimated prob. so that overall averages are the same:
    # Set: $\widetilde{x_i} := c \cdot \frac{x_{i}}{\hat{x}}$
    # where c is the target mean
    #------------------------------------------------------------------
    if df['PROB_POSITIVE'].mean() != natural_positive_prob:         # calculate scaling factor if differing
      c = (natural_positive_prob/(df['PROB_POSITIVE'].mean()))
      df['PROB_POSITIVE'] = c*df['PROB_POSITIVE']               #  rescale posterior values
    
    # Create quantile based Bins where first Bin has observations with the
    # lowest posterior probs:
    #-----------------------------------------------------------------------
    lab=list() ; [lab.append(str(i)) for i in list(np.arange(1,bins+1,1))]
    #[lab.append("Bin "+str(i)) for i in list(np.arange(1,bins+1,1))]            # Make labels for buckets
    
    # Add some white noise to it for uniqueness of the boundaries... 
    df['BIN_POSITIVE'], usedbins = pd.qcut(df['PROB_POSITIVE'] + jitter(len(df['PROB_POSITIVE']), seed = 42), bins, labels = lab,retbins = True, duplicates = duplicates, precision = 5)
    
    pos_group_df = df.groupby('BIN_POSITIVE')

    # Predicted and actual Subro rate in each bin:
    #----------------------------------------------
    lift_positive_act = pos_group_df['ACTUAL'].sum()/pos_group_df['ACTUAL'].count()                      # binned actual rate
    lift_positive_score = pos_group_df['PROB_POSITIVE'].sum()/pos_group_df['PROB_POSITIVE'].count()         # binned posterior
    lift_index_positive = (lift_positive_act/natural_positive_prob)                        # calculate Lift score per bin
    lift_positive_score[lift_positive_score>1] = 1                         # in case rescaling has caused out of range values

    # Make additional quintile buckets,
    # to report the lift score of the last quintile:
    #------------------------------------------------
    q = np.linspace(0,1,bins2+1)
    bb = pd.qcut(df['PROB_POSITIVE'] + jitter(len(df['PROB_POSITIVE']), seed = 42), q = q, labels = False, duplicates = duplicates, precision = 5)
    #bb = pd.qcut(df['PROB_POSITIVE'] , q=q, labels=False,
    #             duplicates=duplicates, precision=5)
    pos_group = df['ACTUAL'].groupby(bb)
    avgPost_20 = pos_group.sum()/pos_group.count()
    lift_index_20 = (avgPost_20/natural_positive_prob)
    
    d=[] ;[d.append(str(int(i*100))+"%") for i in q[1:]]
    lift_index_20.index = d ; lift_index_20 = lift_index_20.rename(columns={0: 'Lift'})
    
    # Bind all together for output and plot:
    #----------------------------------------
    lift_df = pd.DataFrame({'RATE_ACTUAL':lift_positive_act,
                            'LIFT':lift_index_positive,
                            'POSTERIOR': lift_positive_score,
                            'BASELINE_RATE':natural_positive_prob})
    
    # Plot lift chart
    #----------------------------------------------------------------------------------
    if plot_lift:
        
     from plotly.offline import init_notebook_mode, iplot
     import plotly.graph_objs as go

     init_notebook_mode(connected=True)
     if title is None:
        title = 'Predicted vs. actual subro rate (avg. subro rate'+" %.3f" % natural_positive_prob +")"
        
     trace1 = go.Scatter(
            x=lift_df.index,
           y=lift_df['RATE_ACTUAL'],
              name = 'Actual Average Recoveries', xaxis = 'x')

     trace2 = go.Scatter(
           x=lift_df.index,
            y=lift_df['POSTERIOR'],
             name = 'Predicted Average Recoveries', xaxis = 'x')       

     data = [trace1, trace2]
     fig = dict(data=data)

     layout = go.Layout(
           title=title,
           legend=dict(orientation="h"), 
            xaxis=dict(
               title='', #'Binned population',
               titlefont=dict(
                 family='Courier New, monospace',
                   size=18,
                 color='#7f7f7f'
               ),
                tickwidth=4,
                tickangle=45
             ),
              yaxis=dict(
            title='Rate %',
               titlefont=dict(
               family='Courier New, monospace',
             size=18,
                color='#7f7f7f'
            )
     ))
     fig = go.Figure(data=data, layout=layout) ; iplot(fig)   
    #-----------------------------------------------------------------------------    
    return lift_df, total_positive_n, total_n,  round(lift_index_20,3)
#----------------------------------------------------------------------------------------------------------------



class ReduceVIF(BaseEstimator, TransformerMixin):
    
    """Backward feature selection: drop features recursively based in VIF """
    
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # Rule of thump: values between 5 and 10 are deemed "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        #if impute:
        #    self.imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X

    

def entropy(x, **kwargs):
    """Discretize continous 1D variable x and compute entropy"""
    X = deepcopy(x)
    counts, bins  = np.histogram(X, density=False, **kwargs)
    cnoise = counts + jitter(len(counts))   # apply jitter to avoid singularities due to log(0)
    px = cnoise/cnoise.sum()        # mass function
    return -round(np.sum(px*np.log(px)),4)    
    
    
def diff_entropy(x : np.array, x_grid : np.array = None, method : str = 'DPM', 
                 n_mixture_comp = 50, kde_bw : float = None, 
                 MCsim = 10**5, cv_nof_cand = 20, verbose : bool = True):
    """
    2-step estimation of differential entropy of 1D sample of continous rv X.
    First estimate density function f(x) via KDE with CV, DPM or GMM. 
    Then approximate 1D integral via Monte Carlo estimate 
    using random draws from estimate f(x).
    
    Input:
    x: array containg a sample of x ~ F(x)
    """
    
    X = deepcopy(x); output = dict()
    assert method in ['DPM', 'GMM', 'KDE'], 'Method {} not supported!'.format(method)
    
    if method == 'KDE':
        output['bw'] = kde_bw
        if output['bw'] is None:
            if verbose : print('Finding optimal KDE bandwidth using cv.')
            bandwidths = 10 ** np.linspace(-1, 1, cv_nof_cand)     # bw candidates
            grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv = LeaveOneOut())
            grid.fit(X[:, None]);
            output['bw'] = grid.best_params_['bandwidth']

        # Compute density with KDE
        if verbose : print('Estimating kernel density.')
        model = KernelDensity(bandwidth=output['bw'], kernel='gaussian') ; 
        
    # finite_GMM
    if method == 'GMM':    
        if verbose : print('Estimating density via Gaussian mixture model.')
        model = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_distribution", weight_concentration_prior = 1.1, 
            n_components = n_mixture_comp, reg_covar=0, init_params='random',covariance_type = 'diag',
            max_iter=1500, mean_precision_prior=.8,    # vague prior
            random_state=None)
        
    # infinite_GMM    
    if method == 'DPM':
        if verbose : print('Estimating density via Dirichlet process mixture model.')
        model = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process", weight_concentration_prior = 1.1,  
            n_components = n_mixture_comp, reg_covar=0, init_params='random', covariance_type = 'diag',
            max_iter=1500, mean_precision_prior=.8,
            random_state=None)
    
    model.fit(X[:, None])
    
    # In case you want to plot the result, evaluate (x, hat(f(x))):
    if x_grid is not None:
        logprob = model.score_samples(x_grid[:, None])   #log of the probability density
        dens_est_grid = np.exp(logprob)
        output['x'] = x_grid; output['fx'] = dens_est_grid
    
    # Draw random sample from density estimate f(x)
    if method == 'KDE':
        x_sam = model.sample(MCsim)     # sampling for KDE only works with Gaussian kernel, otherwise use mixture approach
    else:
        x_sam, _ = model.sample(MCsim)
    dens_est = np.exp(model.score_samples(x_sam))

    # Monte Carlo estimate of differential entropy using a sample from the estimated f(x):
    if verbose : print('Approximating differential entropy using MC integration.')
    diff_entropy = -np.mean(np.log(dens_est))
    return output, diff_entropy
        
    
def info_crit(X):
    """Compute Bayesian information criterion and Hannan-Quinn for a GEV distribution"""
    x = deepcopy(X) ; k = 3    # number of estimated parameters
    xi_hat, mu_hat, sigma_hat = gev.fit(x)
    log_like = np.sum(gev.logpdf(x, xi_hat, mu_hat, sigma_hat))
    aic = round(2*k -2*log_like,3)
    bic = round(k*np.log(len(x)) -2*log_like,3)
    hq = round(2*k*np.log(np.log(len(x))) -2*log_like,3)
    return aic, bic, hq


def logsumexp(x):
    """Log-Sum-Exp Trick"""
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

    
def model_prob(Scores : pd.DataFrame):
    """
    Calculate quasi model probab. using BIC, 
    motivated through a Laplace approx. of the marginal likelihood
    Input: Dataframe with scores in columns and model names as column names   
    """
    #print("Calculating model fit with respect to GEV(xi, mu, sigma) distribution.")
    prior_M = np.array([1/len(Scores.columns)]*len(Scores.columns))
    min_init = np.inf; winner = None; bic_s = [min_init]; aic_s = [min_init]
    for t, m in enumerate(Scores.columns):
       aic, bic, hq = info_crit(Scores[m].values)   # evaluate fit to a GEV reference
       aic_s.append(aic); bic_s.append(bic)
       if bic < bic_s[t]: winner = m     # for model selection purposes
    x = -0.5*np.array(bic_s[1:]) + np.log(prior_M)
    return {'models': list(Scores.columns),'AIC' : aic_s[1:],'BIC' : bic_s[1:], 'post_model_k' : np.exp(x - logsumexp(x))}    # model probablities    
    
    
    
    
    
        from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
from sklearn.mixture import BayesianGaussianMixture
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import re, functools, logging, re
import pandas as pd
import numpy as np
from pandas.core.base import PandasObject
import pdb       # debugger
from scipy.stats import genextreme as gev
from gensim.parsing.preprocessing import strip_short
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from copy import deepcopy
#import spacy, en_core_web_sm


# Mimics R's paste() function for two lists:
#---------------------------------------------
def reduce_concat(x, sep=""):
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)

def paste(*lists, sep=" ", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)
#----------------------------------------------------


def jitter(M: int, noise_scale: float = 10**5., seed : int = None):
    
  """ Generates jitter that can be added to any float, e.g.
      helps when used with pd.qcut to produce unique class edges
      M: number of random draws, i.e. size
  """  
  if seed is not None:
     np.random.seed(seed)
  return np.random.random(M)/noise_scale ;


def switch(key, switcher = {'years': 'Y', 'days': 'D', 'weeks': 'W','months': "M"}):
        return switcher.get(key, "Key does not match any in 'switcher' argument.")  


class preprocess_text(TransformerMixin):    
   """
   This is an extension of the clean_text class below. 
   Stemming and stopwords filtering were added for preprocessing.
   """ 
   def __init__(self, input_col, output_col, stemming = True, unique_tokens = True):
      self.name = output_col
      self.input_col = input_col
      self.stemming = stemming
      self.unique_tokens = unique_tokens
      if self.unique_tokens : print("Deduplicating tokens per document/sentence.")
      self.regexp = re.compile('(?u)(?:(?!\d)\w)+\\w+')        # remove punctuation and digits
      if self.stemming : self.stemmer = PorterStemmer() ; print("Applying Porter stemming.")
      self.stop_words = set(stopwords.words('english'))

   def fit(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      return self

   def transform(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      self.df_raw = deepcopy(X)
      d = self.df_raw[self.input_col].tolist()
      for i in range(len(d)):
        d[i] = " ".join(self.regexp.findall(str(d[i]))).lower().strip()        # strip ws and convert to lower
        tokens = word_tokenize(d[i])
        if self.unique_tokens:
            # Make tokens unique in sentence
            #tokens = list(set(tokens))   # this loses word order
            my_unique_sentence = []
            e = [my_unique_sentence.append(tok) for tok in tokens if tok not in my_unique_sentence]
            tokens = my_unique_sentence
 
        # Stem words and remove stopwords:
        if self.stemming:
           filtered_tokens = [self.stemmer.stem(w) for w in tokens if not w in self.stop_words]
        else:
           filtered_tokens = [w for w in tokens if not w in self.stop_words]
        d[i] = " ".join(filtered_tokens)
      self.df_raw[self.name] = d 
      return self.df_raw


class clean_text(BaseEstimator, TransformerMixin):
    
   def __init__(self, input_col, output_col):
      self.name = output_col
      self.input_col = input_col
      self.regexp = re.compile('(?u)(?:(?!\d)\w)+\\w+')        # remove punctuation and digits

   def fit(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      return self

   def transform(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      self.df_raw = deepcopy(X)
      d = self.df_raw[self.input_col].tolist()
      for i in range(len(d)):
        d[i] = " ".join(self.regexp.findall(str(d[i]))).lower().strip()        # strip ws and convert to lower
      self.df_raw[self.name] = d 
      return self.df_raw
    
#-------------------------------------------------------------------------------------------------------------------    

class StringCleaner(TransformerMixin):

    def __init__(self, mapping = {},  replace = True, verbose = True):
        # example mapping ={'column_name':{'find':'replacement}}
        self.replace = replace
        #self.convert_cols = convert_cols
        self.mapping = mapping
        self.verbose = verbose
    def fit(self):
        pass
    
    def transform(self, X, minimum_length = 5):
         if isinstance(X, pd.Series):
            return X
         if isinstance(X, pd.DataFrame):
            X_temp = deepcopy(X)
            for col in self.mapping:
                X_temp[col] = X_temp[col].str.lower()
              
            for col in self.mapping:
                for word in self.mapping[col]:
                    X_temp[col] = X_temp[col].str.replace(word,self.mapping[col][word])
            if self.verbose:
                print("Replaced strings in columns: "+ str([col for col in self.mapping]))
            return X_temp
            
"""
def doc_tagger(sentence): 
    
    """'Tag document with linguistic annotations'"""
    
    nlp = en_core_web_sm.load()
    doc = nlp(str(sentence))
    token_text, token_lemma, token_pos, token_tag, token_dep, nounIndices = [], [], [], [], [], []
    index, i = 2*[0]
    annotations = {}
    
    # Linguistic annotations:
    for token in doc:
        token_text.append(token.text) ; token_lemma.append(token.lemma_)
        token_pos.append(token.pos_) ; token_tag.append(token.tag_)
        token_dep.append(token.dep_) ; 
        
        annotations.update({i : [token.lemma_, token.pos_, token.dep_, token.shape_]})
        i += 1
    
    token_char = {'orig token' : token_text, 'pos' : token_pos, 'nounIndices' :nounIndices, 
                  'tag' : token_tag, 'dep': token_dep,  'lemma token' : token_lemma}  
    
    return annotations #, token_char 
"""

#-------------------------------------------------------------------------------------------------------------------    

def get_cases(descr, search_list_noun = ['truck', "car", 'vehicle', 'third party',"third", \
                                         'insured', 'van', 'driver', 'traffic', 'rear'], \
                     search_list_verb = ['drove', 'collided', 'struck', 'hit','parked', "reversed", 'turned'], \
              thresh = 1): 
   """
   Input:
   descr: claims descr. column
   search_list_noun:    search list with nouns to search for in the claims descr
   search_list_verb:    search list with verbs to search for 
   thresh: threshold parameter; overall number of hits, in both nouns and verbs search list 
   """ 
        
   def is_phrase_in(phrase, text):
    """Exact string matcher"""
    return re.search(r"\b{}\b".format(phrase), text, re.IGNORECASE) is not None ;

   weights = {'noun_weight' : 0, 'vrb_weight' : 0}

   for i in search_list_noun:
        if is_phrase_in(i, descr):
          weights['noun_weight'] +=1  
   for i in search_list_verb:
        if is_phrase_in(i, descr):
          weights['vrb_weight'] +=1  
        
   return (weights['noun_weight'] + weights['vrb_weight']) > thresh

    
#----------------------------------------------------------------------------------------------------------------------    

def create_logger(**kwargs):
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
 
    # create the logging file handler
    fh = logging.FileHandler(**kwargs)
 
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
 
    # add handler to logger object
    logger.addHandler(fh)
    return logger

# Set global settings in Jupyter notebook for convenience
#---------------------------------------------------------------------------------------------------------
def start():
    options = {
        'display': {
            'max_columns': 500,
            #'max_colwidth': 25,
            #'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 500,
            #'max_seq_items': 50,         # Max length of printed sequence
            'precision': 4,
            'width' : 1000
            #'show_dimensions': False
        },
        'mode': {
            'chained_assignment': None   # Controls SettingWithCopyWarning
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+

#---------------------------------------------------------------------------------------------------------------

def make_colnames(df, inplace = False):
    """Reformat column names for compatibility with SQl server"""
    regexp = re.compile('(?u)(?:(?!\d)\w)+\\w+')        # remove punctuation and digits
    d = df.columns
    cols = [];
    for i in range(len(d)):
        a = "_".join(regexp.findall(str(d[i]))).lower().strip()       # strip ws and convert to lower; set underscores
        cols.append(a)
    df.columns = cols
    if inplace is not True:
      return df
PandasObject.make_colnames = make_colnames   

#---------------------------------------------------------------------------------------------------------------

class train_test_split_extend(BaseEstimator, TransformerMixin):

  """ Partition the data into training/validation/testing subsets"""

  def __init__(self):
      return None;

  def fit(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      return self

  def transform(self, X, y=None, sizes=[], random_state=None, shuffle=True):

    assert isinstance(sizes, list)
    assert isinstance(X, pd.DataFrame)
    self.X = X
    self.y = y
    self.sizes = sizes
    self.random_state = random_state
    self.shuffle = shuffle

    if y is None: 

        if len(self.sizes) == 3:
          p_rest = round(1-self.sizes[0],2) ; p_test = self.sizes[2]/p_rest
          X_train, X_rest = train_test_split(self.X, test_size = p_rest, random_state = self.random_state, shuffle = self.shuffle)
          X_valid, X_test = train_test_split(X_rest, test_size=p_test, random_state=self.random_state,shuffle=self.shuffle)
          assert (self.X.shape[0] == X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
          return X_train, X_valid, X_test

        if len(self.sizes) == 2:
          p_test = round(1-self.sizes[0],2) ;
          X_train, X_test = train_test_split(self.X, test_size = p_test, random_state=self.random_state,shuffle=self.shuffle)
          assert (self.X.shape[0] == X_train.shape[0] + X_test.shape[0])
          return X_train, X_test

    else:    
        assert isinstance(y, pd.DataFrame)
        if len(self.sizes) == 3:
          p_rest = round(1-self.sizes[0],2) ; p_test = self.sizes[2]/p_rest
          X_train, X_rest, y_train, y_rest = train_test_split(self.X, self.y, stratify = self.y, test_size = p_rest, random_state = self.random_state, shuffle = self.shuffle)
          X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, stratify = y_rest, test_size=p_test, random_state=self.random_state,shuffle=self.shuffle)
          assert (self.X.shape[0] == X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
          return X_train, X_valid, X_test, y_train, y_valid, y_test;

        if len(self.sizes) == 2:
          p_test = round(1-self.sizes[0],2) ;
          X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify = self.y, test_size = p_test, random_state=self.random_state,shuffle=self.shuffle)
          assert (self.X.shape[0] == X_train.shape[0] + X_test.shape[0])
          return X_train, X_test, y_train, y_test;
        

# Calculate and print lift graph:
#-------------------------------------------------------------------------------------------------------------------------------
def calc_lift(y_prob: np.ndarray, y_actual: pd.DataFrame, bins: int = 100, bins2: int = 5, plot_lift = False, duplicates = "raise", title = None):

    """Input:
       y_prob: posterior probs for y=1
       y_actual: true labels
       
       Return:  
       lift_df: True and predicted subro rate, lift value per bin and overall avg. subro rate as baseline
       lift_index_20: Lift indexes for quintiles (20% bins)
       title:   title for plotly lift chart
    """
    y_actual.index = range(len(y_actual))
    cols = ['ACTUAL','PROB_POSITIVE'] 
    data = [y_actual,y_prob] 
    df = pd.DataFrame(dict(zip(cols,data)))
    total_positive_n = df['ACTUAL'].sum()                     
    total_n = len(df.index)                                    #Total Observations
    natural_positive_prob = total_positive_n/float(total_n)         # overall subro rate

    # Normalize estimated prob. so that overall averages are the same:
    # Set: $\widetilde{x_i} := c \cdot \frac{x_{i}}{\hat{x}}$
    # where c is the target mean
    #------------------------------------------------------------------
    if df['PROB_POSITIVE'].mean() != natural_positive_prob:         # calculate scaling factor if differing
      c = (natural_positive_prob/(df['PROB_POSITIVE'].mean()))
      df['PROB_POSITIVE'] = c*df['PROB_POSITIVE']               #  rescale posterior values
    
    # Create quantile based Bins where first Bin has observations with the
    # lowest posterior probs:
    #-----------------------------------------------------------------------
    lab=list() ; [lab.append(str(i)) for i in list(np.arange(1,bins+1,1))]
    #[lab.append("Bin "+str(i)) for i in list(np.arange(1,bins+1,1))]            # Make labels for buckets
    
    # Add some white noise to it for uniqueness of the boundaries... 
    df['BIN_POSITIVE'], usedbins = pd.qcut(df['PROB_POSITIVE'] + jitter(len(df['PROB_POSITIVE']), seed = 42), bins, labels = lab,retbins = True, duplicates = duplicates, precision = 5)
    
    pos_group_df = df.groupby('BIN_POSITIVE')

    # Predicted and actual Subro rate in each bin:
    #----------------------------------------------
    lift_positive_act = pos_group_df['ACTUAL'].sum()/pos_group_df['ACTUAL'].count()                      # binned actual rate
    lift_positive_score = pos_group_df['PROB_POSITIVE'].sum()/pos_group_df['PROB_POSITIVE'].count()         # binned posterior
    lift_index_positive = (lift_positive_act/natural_positive_prob)                        # calculate Lift score per bin
    lift_positive_score[lift_positive_score>1] = 1                         # in case rescaling has caused out of range values

    # Make additional quintile buckets,
    # to report the lift score of the last quintile:
    #------------------------------------------------
    q = np.linspace(0,1,bins2+1)
    bb = pd.qcut(df['PROB_POSITIVE'] + jitter(len(df['PROB_POSITIVE']), seed = 42), q = q, labels = False, duplicates = duplicates, precision = 5)
    #bb = pd.qcut(df['PROB_POSITIVE'] , q=q, labels=False,
    #             duplicates=duplicates, precision=5)
    pos_group = df['ACTUAL'].groupby(bb)
    avgPost_20 = pos_group.sum()/pos_group.count()
    lift_index_20 = (avgPost_20/natural_positive_prob)
    
    d=[] ;[d.append(str(int(i*100))+"%") for i in q[1:]]
    lift_index_20.index = d ; lift_index_20 = lift_index_20.rename(columns={0: 'Lift'})
    
    # Bind all together for output and plot:
    #----------------------------------------
    lift_df = pd.DataFrame({'RATE_ACTUAL':lift_positive_act,
                            'LIFT':lift_index_positive,
                            'POSTERIOR': lift_positive_score,
                            'BASELINE_RATE':natural_positive_prob})
    
    # Plot lift chart
    #----------------------------------------------------------------------------------
    if plot_lift:
        
     from plotly.offline import init_notebook_mode, iplot
     import plotly.graph_objs as go

     init_notebook_mode(connected=True)
     if title is None:
        title = 'Predicted vs. actual subro rate (avg. subro rate'+" %.3f" % natural_positive_prob +")"
        
     trace1 = go.Scatter(
            x=lift_df.index,
           y=lift_df['RATE_ACTUAL'],
              name = 'Actual Average Recoveries', xaxis = 'x')

     trace2 = go.Scatter(
           x=lift_df.index,
            y=lift_df['POSTERIOR'],
             name = 'Predicted Average Recoveries', xaxis = 'x')       

     data = [trace1, trace2]
     fig = dict(data=data)

     layout = go.Layout(
           title=title,
           legend=dict(orientation="h"), 
            xaxis=dict(
               title='', #'Binned population',
               titlefont=dict(
                 family='Courier New, monospace',
                   size=18,
                 color='#7f7f7f'
               ),
                tickwidth=4,
                tickangle=45
             ),
              yaxis=dict(
            title='Rate %',
               titlefont=dict(
               family='Courier New, monospace',
             size=18,
                color='#7f7f7f'
            )
     ))
     fig = go.Figure(data=data, layout=layout) ; iplot(fig)   
    #-----------------------------------------------------------------------------    
    return lift_df, total_positive_n, total_n,  round(lift_index_20,3)
#----------------------------------------------------------------------------------------------------------------



class ReduceVIF(BaseEstimator, TransformerMixin):
    
    """Backward feature selection: drop features recursively based in VIF """
    
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # Rule of thump: values between 5 and 10 are deemed "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        #if impute:
        #    self.imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X

    

def entropy(x, **kwargs):
    """Discretize continous 1D variable x and compute entropy"""
    X = deepcopy(x)
    counts, bins  = np.histogram(X, density=False, **kwargs)
    cnoise = counts + jitter(len(counts))   # apply jitter to avoid singularities due to log(0)
    px = cnoise/cnoise.sum()        # mass function
    return -round(np.sum(px*np.log(px)),4)    
    
    
def diff_entropy(x : np.array, x_grid : np.array = None, method : str = 'DPM', 
                 n_mixture_comp = 50, kde_bw : float = None, 
                 MCsim = 10**5, cv_nof_cand = 20, verbose : bool = True):
    """
    2-step estimation of differential entropy of 1D sample of continous rv X.
    First estimate density function f(x) via KDE with CV, DPM or GMM. 
    Then approximate 1D integral via Monte Carlo estimate 
    using random draws from estimate f(x).
    
    Input:
    x: array containg a sample of x ~ F(x)
    """
    
    X = deepcopy(x); output = dict()
    assert method in ['DPM', 'GMM', 'KDE'], 'Method {} not supported!'.format(method)
    
    if method == 'KDE':
        output['bw'] = kde_bw
        if output['bw'] is None:
            if verbose : print('Finding optimal KDE bandwidth using cv.')
            bandwidths = 10 ** np.linspace(-1, 1, cv_nof_cand)     # bw candidates
            grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv = LeaveOneOut())
            grid.fit(X[:, None]);
            output['bw'] = grid.best_params_['bandwidth']

        # Compute density with KDE
        if verbose : print('Estimating kernel density.')
        model = KernelDensity(bandwidth=output['bw'], kernel='gaussian') ; 
        
    # finite_GMM
    if method == 'GMM':    
        if verbose : print('Estimating density via Gaussian mixture model.')
        model = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_distribution", weight_concentration_prior = 1.1, 
            n_components = n_mixture_comp, reg_covar=0, init_params='random',covariance_type = 'diag',
            max_iter=1500, mean_precision_prior=.8,    # vague prior
            random_state=None)
        
    # infinite_GMM    
    if method == 'DPM':
        if verbose : print('Estimating density via Dirichlet process mixture model.')
        model = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process", weight_concentration_prior = 1.1,  
            n_components = n_mixture_comp, reg_covar=0, init_params='random', covariance_type = 'diag',
            max_iter=1500, mean_precision_prior=.8,
            random_state=None)
    
    model.fit(X[:, None])
    
    # In case you want to plot the result, evaluate (x, hat(f(x))):
    if x_grid is not None:
        logprob = model.score_samples(x_grid[:, None])   #log of the probability density
        dens_est_grid = np.exp(logprob)
        output['x'] = x_grid; output['fx'] = dens_est_grid
    
    # Draw random sample from density estimate f(x)
    if method == 'KDE':
        x_sam = model.sample(MCsim)     # sampling for KDE only works with Gaussian kernel, otherwise use mixture approach
    else:
        x_sam, _ = model.sample(MCsim)
    dens_est = np.exp(model.score_samples(x_sam))

    # Monte Carlo estimate of differential entropy using a sample from the estimated f(x):
    if verbose : print('Approximating differential entropy using MC integration.')
    diff_entropy = -np.mean(np.log(dens_est))
    return output, diff_entropy
        
    
def info_crit(X):
    """Compute Bayesian information criterion and Hannan-Quinn for a GEV distribution"""
    x = deepcopy(X) ; k = 3    # number of estimated parameters
    xi_hat, mu_hat, sigma_hat = gev.fit(x)
    log_like = np.sum(gev.logpdf(x, xi_hat, mu_hat, sigma_hat))
    aic = round(2*k -2*log_like,3)
    bic = round(k*np.log(len(x)) -2*log_like,3)
    hq = round(2*k*np.log(np.log(len(x))) -2*log_like,3)
    return aic, bic, hq


def logsumexp(x):
    """Log-Sum-Exp Trick"""
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

    
def model_prob(Scores : pd.DataFrame):
    """
    Calculate quasi model probab. using BIC, 
    motivated through a Laplace approx. of the marginal likelihood
    Input: Dataframe with scores in columns and model names as column names   
    """
    #print("Calculating model fit with respect to GEV(xi, mu, sigma) distribution.")
    prior_M = np.array([1/len(Scores.columns)]*len(Scores.columns))
    min_init = np.inf; winner = None; bic_s = [min_init]; aic_s = [min_init]
    for t, m in enumerate(Scores.columns):
       aic, bic, hq = info_crit(Scores[m].values)   # evaluate fit to a GEV reference
       aic_s.append(aic); bic_s.append(bic)
       if bic < bic_s[t]: winner = m     # for model selection purposes
    x = -0.5*np.array(bic_s[1:]) + np.log(prior_M)
    return {'models': list(Scores.columns),'AIC' : aic_s[1:],'BIC' : bic_s[1:], 'post_model_k' : np.exp(x - logsumexp(x))}    # model probablities    
    
    
    
    
    
        