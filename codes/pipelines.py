
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from Code.Python.ILDA.Independant_LDA import Independant_LDA
from Code.Python.ILDA.Independant_LDA import LDA as handmade_LDA
from Code.Python.ILDA.Independant_LDA import Independant_LDA_Lasso
from mne.decoding import CSP


# Définissez le banc de filtres (filter bank) à utiliser







def get_pipeline():
    pipelines = {}
    #filters = [[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]]
    # pipelines['fb_bpow+lda'] =make_pipeline(FilterBank(LogVariance()), LDA())
    # pipelines['fb_bpow+handmade_lda'] = make_pipeline(FilterBank(LogVariance()), handmade_LDA())
    # pipelines['fb_bpow+lda_cov'] = make_pipeline(FilterBank(LogVariance()), handmade_LDA())
    # pipelines['fb_bpow+Ilda'] =make_pipeline(FilterBank(LogVariance()), Independant_LDA())
    # -----------------------------------------------------------------------
    pipelines['6 csp+lda'] = make_pipeline(CSP(n_components=6), LDA())
    # pipelines['6 csp+handmade_lda'] = make_pipeline(CSP(n_components=6), handmade_LDA())
    # pipelines['6 csp+Indedpendant_lda_lasso'] = make_pipeline(CSP(n_components=6), Independant_LDA_Lasso())
    # pipelines['6 csp+Indedpendant_lda'] = make_pipeline(CSP(n_components=6), Independant_LDA())

    # pipelines['6 csp+handmade_lda'] = make_pipeline(CSP(n_components=6), handmade_LDA())
    # pipelines['6 csp+lda_cov'] = make_pipeline(CSP(n_components=6), handmade_LDA())
    pipelines['6 csp+Indedpendant_lda'] = make_pipeline(CSP(n_components=6), Independant_LDA())
    # pipelines['8 csp+lda'] = make_pipeline(CSP(n_components=8), handmade_LDA())
    # pipelines['8 csp+lda_cov'] = make_pipeline(CSP(n_components=8), handmade_LDA())
    # # # # pipelines['6 csp+handmade_lda'] = make_pipeline(CSP(n_components=6), handmade_LDA())
    # pipelines['8 csp+Indedpendant_lda'] = make_pipeline(CSP(n_components=8), Independant_LDA())
    # # pipelines['6 csp+Indedpendant_lda'] = make_pipeline(CSP(n_components=6), Independant_LDA())
    # # pipelines['xdawn+svm'] = make_pipeline(Xdawn(nfilter=4), LDA())
    # pipelines['tgsp+svm'] = make_pipeline(Covariances(estimator='lwf'),
    #                                       TangentSpace(metric='riemann'),
    #                                       SVC(kernel='linear'))
    # # pipelines['MDM'] = make_p
    #
    # pipelines['XDAWN+tgsp+svm'] = make_pipeline(XdawnCovariances(estimator='lwf'),
    #                                       TangentSpace(metric='riemann'),
    #                                       SVC(kernel='linear'))
    # pipelines['XDAWN+MDM'] = make_pipeline(XdawnCovariances(estimator='lwf'),
    #                                  MDM(metric='riemann', n_jobs=-1))

    # pipelines['FBSCP+ lda'] =make_pipeline(fbcsp(), LDA())
    return pipelines


#%%
