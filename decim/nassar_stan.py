import numpy as np

from decim import glaze2 as gl

# no prior so far


def nassar_model():
    '''
    Returns stan modelling code for glaze model for multiple blocks in one session.
    '''
    nassar = """
        functions {
            real tau(real cpp, real tau, real mu, real x){
                real a;
                real c;
                a = (25 * cpp + (1-cpp) * tau * 25 + cpp * (1-cpp) * (tau + mu * (1 - tau) - x));
                c = a + 25;
                return (a / c);
            }
        }
        data {
            int<lower=1> B; // number of blocks
            int<lower=1> b[B+1]; // integer array with indices of last point locations of block
            int<lower=1> I; // number of trials
            vector[I] y; // subjects answer
            vector[I] x; //samples
        }
        transformed data {
            real Nsq;
            real Ux;
            Nsq= 5.;
            Ux = 1. / 300;
        }
        parameters {
            real<lower=.001, upper=1> H;
            real<lower=0> sig;
            real<lower=0, upper=1> lap; //Lapse rate
        }
        transformed parameters {
            vector[I+1] mu; // estimated mean
            vector[I] alpha; // learning rate
            vector[I] delta; // prediction error
            vector[I] cpp; // change point probability
            vector[I+1] t; // relative uncertainty tau
            vector[I] sigma; // variance of predictive distribution
            vector[I] Nx; // predictive distribution if cp did not occur

            for (i in 1:B){
                mu[b[i]] = x[b[i]];
                t[b[i]] = .5;
                for (n in (b[i]):(b[i+1]-1)){
                    sigma[n] = (square(Nsq) + (t[n] * square(Nsq) / (1 - t[n])));
                    Nx[n] = normal_lpdf(x[n] | mu[n], sigma[n]);
                    cpp[n] = (Ux * H) / (Ux * H + exp(Nx[n]) * (1 - H));
                    alpha[n] = cpp[n] * (1 - t[n]) + t[n];
                    delta[n] = x[n] - mu[n];
                    t[n+1] = tau(cpp[n], t[n], mu[n], x[n]);
                    mu[n+1] = mu[n] + alpha[n] * delta[n];
                }
            }
        }
        model {
            lap ~ beta(1.4, 8.6);
            H ~ uniform(0,1);
            sig ~ gamma(1,1);
            for (j in 1:I) {

                target += log_mix(lap, uniform_lpdf(y[j] | 0,300),
                                   normal_lpdf(y[j] | mu[j+1], sig));

            }
        }
        """
    return nassar


def nassar_data(subject, session, phase, blocks, path, swap=False):
    '''
    Returns dictionary with data that fits requirement of stan model.

    Takes subject, session, phase, list of blocks and filepath.
    '''
    lp = [1]
    X = []
    y = []
    for block in blocks:
        sourcepd = gl.log2pd(gl.load_log(subject, session, phase, block, path), block)
        predictions = sourcepd.loc[sourcepd.message == 'PRD_PRED'].value.values.astype(float)
        samples = sourcepd.loc[sourcepd.message == 'PRD_PREDICT_SAMPLE_ON'].value.values.astype(float)
        trials = len(samples)
        lp.append(trials)
        X.append(list(samples))
        y.append(list(predictions))
    x = sum(X, [])
    y = sum(y, [])
    assert len(x) == len(y)
    data = {
        'I': len(x),
        'B': len(blocks),
        'b': np.cumsum(lp),
        'x': np.array(x),
        'y': np.array(y)
    }
    return data


__version__ = '1.0.1'
'''
not working completely yet...
'''

#print(nassar_data('VPIM04', 'B', 4, [1, 2, 3, 4, 5, 6, 7], '/Users/kenohagena/Documents/immuno/data/vaccine'))
