project{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3M\n",
      "American Express\n",
      "Apple\n",
      "Boeing\n",
      "Caterpillar\n",
      "Chevron\n",
      "Cisco Systems\n",
      "Coca-Cola\n",
      "DowDuPont\n",
      "ExxonMobil\n",
      "General Electric\n",
      "Goldman Sachs\n",
      "The Home Depot\n",
      "IBM\n",
      "Intel\n",
      "Johnson & Johnson\n",
      "JPMorgan Chase\n",
      "McDonald's\n",
      "Merck\n",
      "Microsoft\n",
      "Nike\n",
      "Pfizer\n",
      "Procter & Gamble\n",
      "Travelers\n",
      "UnitedHealth Group\n",
      "United Technologies\n",
      "Verizon\n",
      "Visa\n",
      "Walmart\n",
      "Walt Disney\n",
      "['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DWDP', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'VZ', 'V', 'WMT', 'DIS']\n",
      "Walt Disney\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['MMM',\n",
       " 'AXP',\n",
       " 'AAPL',\n",
       " 'BA',\n",
       " 'CAT',\n",
       " 'CVX',\n",
       " 'CSCO',\n",
       " 'KO',\n",
       " 'DWDP',\n",
       " 'XOM',\n",
       " 'GE',\n",
       " 'GS',\n",
       " 'HD',\n",
       " 'IBM',\n",
       " 'INTC',\n",
       " 'JNJ',\n",
       " 'JPM',\n",
       " 'MCD',\n",
       " 'MRK',\n",
       " 'MSFT',\n",
       " 'NKE',\n",
       " 'PFE',\n",
       " 'PG',\n",
       " 'TRV',\n",
       " 'UNH',\n",
       " 'UTX',\n",
       " 'VZ',\n",
       " 'V',\n",
       " 'WMT',\n",
       " 'DIS']"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "import bs4 as bs\n",
    "from bs4 import BeautifulSoup\n",
    "import pickle\n",
    "import requests\n",
    "import datetime as dt\n",
    "import os\n",
    "import pandas as pd\n",
    "import pandas_datareader.data as web\n",
    "import fix_yahoo_finance as yf\n",
    "yf.pdr_override() \n",
    "\n",
    "def getdowtable():\n",
    "    tickers=[]\n",
    "    name=[]\n",
    "    page = requests.get('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')\n",
    "    pagesoup = BeautifulSoup(page.text,'lxml')\n",
    "    table = pagesoup.find(\"table\",{\"class\":\"wikitable sortable\"})\n",
    "    \n",
    "#     print(table.findAll('tr')[1])\n",
    "    for row in table.findAll('tr')[1:]:\n",
    "        name = row.findAll('td')[0].text\n",
    "        ticker = row.findAll('td')[2].text\n",
    "        mapping = str.maketrans(\".\",\"-\")\n",
    "        ticker = ticker.translate(mapping)\n",
    "        tickers.append(ticker)\n",
    "        print(name)  \n",
    "    with open(\"DowJones.pickle\",\"wb\") as f:\n",
    "        pickle.dump(tickers,f)\n",
    "        print(tickers)\n",
    "        print(name)\n",
    "        return (tickers)\n",
    "getdowtable() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "already have MMM\n",
      "already have AXP\n",
      "already have AAPL\n",
      "already have BA\n",
      "already have CAT\n",
      "already have CVX\n",
      "already have CSCO\n",
      "already have KO\n",
      "already have DWDP\n",
      "already have XOM\n",
      "already have GE\n",
      "already have GS\n",
      "already have HD\n",
      "already have IBM\n",
      "already have INTC\n",
      "already have JNJ\n",
      "already have JPM\n",
      "already have MCD\n",
      "already have MRK\n",
      "already have MSFT\n",
      "already have NKE\n",
      "already have PFE\n",
      "already have PG\n",
      "already have TRV\n",
      "already have UNH\n",
      "already have UTX\n",
      "already have VZ\n",
      "already have V\n",
      "already have WMT\n",
      "already have DIS\n"
     ]
    }
   ],
   "source": [
    "def getdatafromweb(reloaddb=False):\n",
    "    if reloaddb:\n",
    "        tickers=getdowtable() \n",
    "    else:\n",
    "        with open(\"DowJones.pickle\",\"rb\") as f:\n",
    "            tickers=pickle.load(f)\n",
    "            \n",
    "    start= dt.datetime(2000,1,1)\n",
    "    end= dt.datetime(2018,2,28)\n",
    "    \n",
    "    if not os.path.exists('Stockdb'):\n",
    "        os.makedirs('Stockdb')\n",
    "    for ticker in tickers:\n",
    "        if not os.path.exists('Stockdb/{}.csv'.format(ticker)):\n",
    "            df = web.get_data_yahoo('MMM', start, end)\n",
    "            df.to_csv('Stockdb/MMM.csv'.format(ticker))\n",
    "        else:\n",
    "            print(\"already have {}\".format(ticker))\n",
    "getdatafromweb()\n",
    "\n",
    "     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk\n",
    "import pickle\n",
    "import random\n",
    "from nltk.classify.scikitlearn import SklearnClassifier\n",
    "from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB\n",
    "from sklearn.linear_model import LogisticRegression, SGDClassifier\n",
    "from sklearn.svm import SVC, LinearSVC, NuSVC\n",
    "from nltk.classify import ClassifierI\n",
    "from statistics import mode\n",
    "from nltk.tokenize import word_tokenize \n",
    "import codecs\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#naive bayes\n",
    "\n",
    "\n",
    "class Vote(ClassifierI):\n",
    "    def __init__(self, *classifiers):\n",
    "        self._classifiers = classifiers\n",
    "\n",
    "    def classify(self, features):\n",
    "        votes = []\n",
    "        for c in self._classifiers:\n",
    "            v = c.classify(features)\n",
    "            votes.append(v)\n",
    "        return mode(votes)\n",
    "\n",
    "    def confidence(self, features):\n",
    "        votes = []\n",
    "        for c in self._classifiers:\n",
    "            v = c.classify(features)\n",
    "            votes.append(v)\n",
    "\n",
    "        choice_votes = votes.count(mode(votes))\n",
    "        conf = choice_votes / len(votes)\n",
    "        return conf\n",
    "    \n",
    "    \n",
    "short_pos=open('/Users/yash/Downloads/shortpos.txt', 'r').read()\n",
    "short_neg=open('/Users/yash/Downloads/shortneg.txt', 'r').read()\n",
    "\n",
    "\n",
    "documents = []\n",
    "\n",
    "for r in short_pos.split('\\n'):\n",
    "    documents.append( (r, \"pos\") )\n",
    "\n",
    "for r in short_neg.split('\\n'):\n",
    "    documents.append( (r, \"neg\") )\n",
    "    \n",
    "\n",
    "\n",
    "# short_pos = codecs.open(\"/Users/yash/Downloads/shortpos.txt\",\"r\", encoding='latin2').read()\n",
    "# short_neg = codecs.open(\"/Users/yash/Downloads/shortneg.txt\",\"r\",encoding='latin2').read()\n",
    "# short_pos = [line.rstrip('\\n') for line in open('/Users/yash/Downloads/shortpos.txt', 'r', encoding='ISO-8859-1')]\n",
    "# short_neg = [line.rstrip('\\n') for line in open('/Users/yash/Downloads/shortneg.txt', 'r', encoding='ISO-8859-1')]\n",
    "\n",
    "\n",
    "\n",
    "# for r in (short_pos):\n",
    "#     r.split('/n')\n",
    "#     documents.append((r,'pos'))\n",
    "# for r in short_neg:\n",
    "#     r.split('/n')\n",
    "#     documents.append((r,'neg'))\n",
    "    \n",
    "allwords=[]\n",
    "allowed_word_types = [\"J\"]\n",
    "\n",
    "for p in short_pos.split('\\n'):\n",
    "    documents.append( (p, \"pos\") )\n",
    "    words = word_tokenize(p)\n",
    "    pos = nltk.pos_tag(words)\n",
    "    for w in pos:\n",
    "        if w[1][0] in allowed_word_types:\n",
    "            all_words.append(w[0].lower())\n",
    "\n",
    "    \n",
    "for p in short_neg.split('\\n'):\n",
    "    documents.append( (p, \"neg\") )\n",
    "    words = word_tokenize(p)\n",
    "    pos = nltk.pos_tag(words)\n",
    "    for w in pos:\n",
    "        if w[1][0] in allowed_word_types:\n",
    "            all_words.append(w[0].lower())\n",
    "\n",
    "\n",
    "\n",
    "save_documents = open(\"/Users/yash/Downloads/documents.pickle\",\"wb\")\n",
    "pickle.dump(documents, save_documents)\n",
    "save_documents.close()\n",
    "\n",
    "pos_words = word_tokenize(short_pos)\n",
    "neg_words = word_tokenize(short_neg)\n",
    "\n",
    "for w in pos_words:\n",
    "    allwords.append(w.lower())\n",
    "    \n",
    "for w in neg_words:\n",
    "    allwords.append(w.lower())\n",
    "    \n",
    "\n",
    "    \n",
    "allwords= nltk.FreqDist(allwords)\n",
    "word_features= list(allwords.keys())[:5000]\n",
    "\n",
    "save_word_features = open(\"/Users/yash/Downloads/word_features5k.pickle\",\"wb\")\n",
    "pickle.dump(word_features, save_word_features)\n",
    "save_word_features.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "def find_features(document):\n",
    "    words = word_tokenize(document)\n",
    "    features = {}\n",
    "    for w in word_features:\n",
    "        features[w] = (w in words)\n",
    "\n",
    "    return features \n",
    "\n",
    "\n",
    "featuresets = [(find_features(rev), category) for (rev, category) in documents]\n",
    "\n",
    "random.shuffle(featuresets)\n",
    "\n",
    "\n",
    "training = featuresets[:10000]\n",
    "testing =  featuresets[10000:]\n",
    "\n",
    "# save_classifier = open(\"naive_bayes.pickle\",\"rb\")\n",
    "# classifier= pickle.load(save_classifier)\n",
    "# save_classifier.close()\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "# print(training[:10])\n",
    "# # print(testing[:10])\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "original NB classifier accuracy= 71.57894736842105\n",
      "Most Informative Features\n",
      "              engrossing = True              pos : neg    =     20.9 : 1.0\n",
      "                  stupid = True              neg : pos    =     18.5 : 1.0\n",
      "                powerful = True              pos : neg    =     17.7 : 1.0\n",
      "                captures = True              pos : neg    =     17.6 : 1.0\n",
      "               inventive = True              pos : neg    =     13.6 : 1.0\n",
      "              refreshing = True              pos : neg    =     12.9 : 1.0\n",
      "               wonderful = True              pos : neg    =     12.5 : 1.0\n",
      "                    warm = True              pos : neg    =     11.7 : 1.0\n",
      "            refreshingly = True              pos : neg    =     11.6 : 1.0\n",
      "                  deftly = True              pos : neg    =     10.9 : 1.0\n",
      "                    ages = True              pos : neg    =     10.9 : 1.0\n",
      "                   vivid = True              pos : neg    =     10.9 : 1.0\n",
      "                provides = True              pos : neg    =     10.5 : 1.0\n",
      "                portrait = True              pos : neg    =     10.3 : 1.0\n",
      "             examination = True              pos : neg    =     10.1 : 1.0\n"
     ]
    }
   ],
   "source": [
    "classifier = nltk.NaiveBayesClassifier.train(training)\n",
    "print(\"original NB classifier accuracy=\", (nltk.classify.accuracy(classifier,testing))*100)\n",
    "classifier.show_most_informative_features(15)\n",
    "\n",
    "save_classifier = open(\"/Users/yash/Downloads/originalnaivebayes5k.pickle\",\"wb\")\n",
    "pickle.dump(classifier, save_classifier)\n",
    "save_classifier.close()\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "#error in gaussian\n",
    "# GaussianNB_Classifier = SklearnClassifier(GaussianNB())\n",
    "# GaussianNB_Classifier.train(training)\n",
    "# print(\"GaussianNB classifier accuracy=\", (nltk.classify.accuracy(GaussianNB_Classifier,testing))*100)\n",
    "\n",
    "\n",
    "\n",
    "# LogisticRegression, SGDClassifier, SVC, LinearSVC, NuSVC\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "    \n",
    "    \n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MNB classifier accuracy= 69.92481203007519\n"
     ]
    }
   ],
   "source": [
    "MNB_Classifier = SklearnClassifier(MultinomialNB())\n",
    "MNB_Classifier.train(training)\n",
    "print(\"MNB classifier accuracy=\", (nltk.classify.accuracy(MNB_Classifier,testing))*100)\n",
    "save_classifier = open(\"/Users/yash/Downloads/MNB_Classifier5k.pickle\",\"wb\")\n",
    "pickle.dump(MNB_Classifier, save_classifier)\n",
    "save_classifier.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "BernoulliNB classifier accuracy= 70.67669172932331\n"
     ]
    }
   ],
   "source": [
    "\n",
    "BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())\n",
    "BernoulliNB_Classifier.train(training)\n",
    "print(\"BernoulliNB classifier accuracy=\", (nltk.classify.accuracy(BernoulliNB_Classifier,testing))*100)\n",
    "\n",
    "save_classifier = open(\"pickled_algos/BernoulliNB_Classifier5k.pickle\",\"wb\")\n",
    "pickle.dump(BernoulliNB_Classifier, save_classifier)\n",
    "save_classifier.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LogisticRegression classifier accuracy= 71.57894736842105\n"
     ]
    }
   ],
   "source": [
    "LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())\n",
    "LogisticRegression_Classifier.train(training)\n",
    "print(\"LogisticRegression classifier accuracy=\", (nltk.classify.accuracy(LogisticRegression_Classifier,testing))*100)\n",
    "\n",
    "save_classifier = open(\"/Users/yash/Downloads/LogisticRegression_Classifier5k.pickle\",\"wb\")\n",
    "pickle.dump(LogisticRegression_Classifier, save_classifier)\n",
    "save_classifier.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/yash/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.\n",
      "  \"and default tol will be 1e-3.\" % type(self), FutureWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SGDClassifier classifier accuracy= 68.87218045112782\n"
     ]
    }
   ],
   "source": [
    "\n",
    "SGDClassifier_Classifier = SklearnClassifier(SGDClassifier())\n",
    "SGDClassifier_Classifier.train(training)\n",
    "print(\"SGDClassifier classifier accuracy=\", (nltk.classify.accuracy(SGDClassifier_Classifier,testing))*100)\n",
    "\n",
    "\n",
    "save_classifier = open(\"/Users/yash/Downloads/SGDC_Classifier5k.pickle\",\"wb\")\n",
    "pickle.dump(SGDClassifier_Classifier, save_classifier)\n",
    "save_classifier.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC classifier accuracy= 47.51879699248121\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "SVC_Classifier = SklearnClassifier(SVC())\n",
    "SVC_Classifier.train(training)\n",
    "print(\"SVC classifier accuracy=\", (nltk.classify.accuracy(SVC_Classifier,testing))*100)\n",
    "\n",
    "save_classifier = open(\"/Users/yash/Downloads/SVC_Classifier5k.pickle\",\"wb\")\n",
    "pickle.dump(SVC_Classifier, save_classifier)\n",
    "save_classifier.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LinearSVC classifier accuracy= 70.37593984962406\n"
     ]
    }
   ],
   "source": [
    "\n",
    "LinearSVC_Classifier = SklearnClassifier(LinearSVC())\n",
    "LinearSVC_Classifier.train(training)\n",
    "print(\"LinearSVC classifier accuracy=\", (nltk.classify.accuracy(LinearSVC_Classifier,testing))*100)\n",
    "\n",
    "save_classifier = open(\"/Users/yash/Downloads/LinearSVC_Classifier5k.pickle\",\"wb\")\n",
    "pickle.dump(LinearSVC_Classifier, save_classifier)\n",
    "save_classifier.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "NuSVC classifier accuracy= 73.23308270676692\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "\n",
    "NuSVC_Classifier = SklearnClassifier(NuSVC())\n",
    "NuSVC_Classifier.train(training)\n",
    "print(\"NuSVC classifier accuracy=\", (nltk.classify.accuracy(NuSVC_Classifier,testing))*100)\n",
    "\n",
    "save_classifier = open(\"/Users/yash/Downloads/NuSVC_Classifier5k.pickle\",\"wb\")\n",
    "pickle.dump(NuSVC_Classifier, save_classifier)\n",
    "save_classifier.close()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "ename": "NotImplementedError",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNotImplementedError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-22-46135c40dc24>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      7\u001b[0m                         \u001b[0mLinearSVC_Classifier\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      8\u001b[0m                         NuSVC_Classifier)\n\u001b[0;32m----> 9\u001b[0;31m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Voted classifier accuracy=\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mnltk\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclassify\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0maccuracy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mvoted_classifier\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mtesting\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0;36m100\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     10\u001b[0m \u001b[0;31m# print(\"Classification:\", voted_classifer.classify(testing[0][0]), \"cofidence %:\", voted_classifier.confidence(testing[0][0])\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0;31m# print(\"Classification:\", voted_classifer.classify(testing[1][0]), \"cofidence %:\", voted_classifier.confidence(testing[1][0])\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.6/site-packages/nltk/classify/util.py\u001b[0m in \u001b[0;36maccuracy\u001b[0;34m(classifier, gold)\u001b[0m\n\u001b[1;32m     85\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     86\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0maccuracy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mclassifier\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgold\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 87\u001b[0;31m     \u001b[0mresults\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mclassifier\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclassify_many\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mfs\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mfs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0ml\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mgold\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     88\u001b[0m     \u001b[0mcorrect\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0ml\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0mr\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0ml\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mr\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mgold\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresults\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     89\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mcorrect\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.6/site-packages/nltk/classify/api.py\u001b[0m in \u001b[0;36mclassify_many\u001b[0;34m(self, featuresets)\u001b[0m\n\u001b[1;32m     75\u001b[0m         \u001b[0;34m:\u001b[0m\u001b[0mrtype\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mlist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlabel\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     76\u001b[0m         \"\"\"\n\u001b[0;32m---> 77\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclassify\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfs\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mfs\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mfeaturesets\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     78\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     79\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mprob_classify_many\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfeaturesets\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.6/site-packages/nltk/classify/api.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m     75\u001b[0m         \u001b[0;34m:\u001b[0m\u001b[0mrtype\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mlist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlabel\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     76\u001b[0m         \"\"\"\n\u001b[0;32m---> 77\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclassify\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfs\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mfs\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mfeaturesets\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     78\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     79\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mprob_classify_many\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfeaturesets\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.6/site-packages/nltk/classify/api.py\u001b[0m in \u001b[0;36mclassify\u001b[0;34m(self, featureset)\u001b[0m\n\u001b[1;32m     54\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclassify_many\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mfeatureset\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     55\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 56\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mNotImplementedError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     57\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     58\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mprob_classify\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfeatureset\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNotImplementedError\u001b[0m: "
     ]
    }
   ],
   "source": [
    "\n",
    "open_file = open(\"/Users/yash/Downloads/originalnaivebayes5k.pickle\", \"rb\")\n",
    "classifier = pickle.load(open_file)\n",
    "open_file.close()\n",
    "\n",
    "\n",
    "open_file = open(\"/Users/yash/Downloads/MNB_Classifier5k.pickle\", \"rb\")\n",
    "MNB_Classifier = pickle.load(open_file)\n",
    "open_file.close()\n",
    "\n",
    "\n",
    "\n",
    "open_file = open(\"/Users/yash/Downloads/BernoulliNB_Classifier5k.pickle\", \"rb\")\n",
    "BernoulliNB_Classifier = pickle.load(open_file)\n",
    "open_file.close()\n",
    "\n",
    "\n",
    "open_file = open(\"/Users/yash/Downloads/LogisticRegression_Classifier5k.pickle\", \"rb\")\n",
    "LogisticRegression_Classifier = pickle.load(open_file)\n",
    "open_file.close()\n",
    "\n",
    "\n",
    "open_file = open(\"/Users/yash/Downloads/LinearSVC_Classifier5k.pickle\", \"rb\")\n",
    "LinearSVC_Classifier = pickle.load(open_file)\n",
    "open_file.close()\n",
    "\n",
    "open_file = open(\"/Users/yash/Downloads/NuSVC_Classifier5k.pickle\",\"wb\")\n",
    "NuSVC_Classifier=LinearSVC_Classifier = pickle.load(open_file)\n",
    "save_classifier.close()\n",
    "\n",
    "\n",
    "open_file = open(\"/Users/yash/Downloads/SGDC_Classifier5k.pickle\", \"rb\")\n",
    "SGDClassifier_Classifier = pickle.load(open_file)\n",
    "open_file.close()\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "voted_classifier accuracy percent: 72.18045112781954\n"
     ]
    }
   ],
   "source": [
    "voted_classifier =           Vote(classifier,\n",
    "                                  NuSVC_Classifier,\n",
    "                                  LinearSVC_Classifier,\n",
    "                                  MNB_Classifier,\n",
    "                                  BernoulliNB_Classifier,\n",
    "                                  SGDClassifier_Classifier,\n",
    "                                  LogisticRegression_Classifier)\n",
    "\n",
    "print(\"voted_classifier accuracy percent:\", (nltk.classify.accuracy(voted_classifier, testing))*100)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sentiment(text):\n",
    "    feats = find_features(text)\n",
    "    return voted_classifier.classify(feats),voted_classifier.confidence(feats)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
