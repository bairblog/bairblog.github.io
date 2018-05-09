// Entry point for interactive diagrams.
function main() {

  // Put interactive diagrams in their places
  singleHistogramTable();
  singleHistogramInteractiveTable();
  comparisonTable();
  comparisonCurvesTable();
  singleCurvesTable();
  // Set values of true positives vs. false positives.
  var tprValue = 1;
  var fprValue = -4;
  var tprOutcome = 75;
  var fprOutcome = -150;

  var group_ratio = 0.25;
  var base_pop_size = 700;

  // // Parameters for main model comparison.
  // var s0 = 10; // standard deviations of defaulters/payers.
  // var s1 = 10;
  // var d0 = 8;  // differences from means of defaulters/payers
  // var d1 = 12;
  // var m0 = 55; // means of overall distributions
  // var m1 = 45;

  // distributions from pi and rho
  dist0 = getNormalPis(45, 20)
  pi0 = dist0[0], repay_prob0 = dist0[1]
  dist1 = getNormalPis(65, 20)
  pi1 = dist1[0], repay_prob1 = dist1[1]

  // [pi0, pi1, repay_prob0, repay_prob1] = getFICOEmpPis();
  // repay_prob1 = repay_prob0
  // [pi0, pi1, repay_prob0] = getFICOPis();
  // repay_prob1 = repay_prob0;

  // Create items to classify: two groups with different
  // distributions of positive/negative examples.
  // comparisonExample0_nocurves = new GroupModel(makeNormalItems(0, 1, 100, m0 + d0, s0)
  //    .concat(makeNormalItems(0, 0, 100, m0 - d0, s0)), tprValue, fprValue, tprOutcome, fprOutcome);
  // comparisonExample1_nocurves = new GroupModel(makeNormalItems(1, 1, 100, m1 + d1, s1)
  //    .concat(makeNormalItems(1, 0, 100, m1 - d1, s1)), tprValue, fprValue, tprOutcome, fprOutcome);
  // comparisonExample0 = new GroupModel(makeNormalItems(0, 1, 100, m0 + d0, s0)
  //    .concat(makeNormalItems(0, 0, 100, m0 - d0, s0)), tprValue, fprValue, tprOutcome, fprOutcome);
  // comparisonExample1 = new GroupModel(makeNormalItems(1, 1, 100, m1 + d1, s1)
  //    .concat(makeNormalItems(1, 0, 100, m1 - d1, s1)), tprValue, fprValue, tprOutcome, fprOutcome);

  var num_items0 = base_pop_size * group_ratio;
  var num_items1 = base_pop_size * (1-group_ratio);

  var comparisonExample0_nocurves = new GroupModel(makeItems(0, pi0, repay_prob0, num_items0), tprValue, fprValue, tprOutcome, fprOutcome);
  var comparisonExample1_nocurves = new GroupModel(makeItems(1, pi1, repay_prob1, num_items1), tprValue, fprValue, tprOutcome, fprOutcome);
  var comparisonExample0 = new GroupModel(makeItems(0, pi0, repay_prob0, num_items0), tprValue, fprValue, tprOutcome, fprOutcome);
  var comparisonExample1 = new GroupModel(makeItems(1, pi1, repay_prob1, num_items1), tprValue, fprValue, tprOutcome, fprOutcome);

  comparisonExample0.link_setting = 'max-profit'
  comparisonExample1.link_setting = 'max-profit'

  comparisonExample0.setMaps();
  comparisonExample1.setMaps();

  

  // var singleModel = new GroupModel(makeNormalItems(0, 1, 100, 60, 10)
  //     .concat(makeNormalItems(0, 0, 100, 40, 10)), tprValue, fprValue, tprOutcome, fprOutcome);
  // dist = getNormalPis(45, 20)
  var pi = []; var repay_prob = [];
  for (var i in pi0) { 
    // pi.push(group_ratio * pi0[i] + (1-group_ratio) * pi1[i]); 
    pi.push(pi0[i]); 
    // repay_prob.push(group_ratio * repay_prob1[i] + (1-group_ratio) * repay_prob0[i]);
    repay_prob.push(repay_prob0[i]);
  }
  
  var singleModel = new GroupModel(makeItems(2, pi, repay_prob, 0.75 * base_pop_size), tprValue, fprValue, tprOutcome, fprOutcome);
  //var singleModel = new GroupModel(getHardcodedSingleModel(2), tprValue, fprValue, tprOutcome, fprOutcome);
  var singleModel_nocurves = new GroupModel(makeItems(2, pi, repay_prob, 0.75 * base_pop_size), tprValue, fprValue, tprOutcome, fprOutcome);
  //var singleModel_nocurves = new GroupModel(getHardcodedSingleModel(2), tprValue, fprValue, tprOutcome, fprOutcome);
  var singleModel_interactive = new GroupModel(makeItems(2, pi, repay_prob, 0.75 * base_pop_size), tprValue, fprValue, tprOutcome, fprOutcome);
  //var singleModel_interactive = new GroupModel(getHardcodedSingleModel(2), tprValue, fprValue, tprOutcome, fprOutcome);
  singleModel_interactive.setMaps();

  // Make models to represent different distributions.
  //var distributionExample0 = new GroupModel(makeNormalItems(0, 1, 150, 70, 7)
  //    .concat(makeNormalItems(0, 0, 150, 30, 7)), tprValue, fprValue);
  

  // Need to classify to get colors to look right on histogram.
  singleModel.classify(0);

  // Create optimizer for models.
  var optimizer = Optimizer(comparisonExample0, comparisonExample1, 1);

  // Buttons to activate different classification strategies.
  document.getElementById('max-profit').onclick = optimizer.maximizeProfit;
                //set_link_mode('max-profit', [comparisonExample0, comparisonExample1], optimizer); // optimizer.maximizeProfit;
  document.getElementById('demographic-parity').onclick = optimizer.demographicParity
                //set_link_mode('demographic-parity', [comparisonExample0, comparisonExample1], optimizer); 
  document.getElementById('equal-opportunity').onclick = optimizer.equalOpportunity;
                //set_link_mode('equal-opportunity', [comparisonExample0, comparisonExample1], optimizer) 
      //optimizer.equalOpportunity;

  // Make loan matrix
  createLoanMatrix('single-loans0', singleModel_nocurves);
  createLoanMatrix('single-loans1', singleModel_interactive);

  // Make histograms.
  createHistogram('single-histogram', singleModel, true);
  createHistogram('single-histogram0', singleModel_nocurves);
  createHistogram('single-histogram1', singleModel_interactive);

  // Link the 0 and 1 histograms
  createHistogram('histogram0', comparisonExample0, false, true, comparisonExample1);
  createHistogram('histogram1', comparisonExample1, false, true, comparisonExample0);


  createHistogram('histogram0_nocurves', comparisonExample0_nocurves, false, true);
  createHistogram('histogram1_nocurves', comparisonExample1_nocurves, false, true);

  // Add legends.
  createSimpleHistogramLegend('single-histogram-legend', 2);
  createHistogramLegend('single-histogram-legend0', 2);
  createHistogramLegend('single-histogram-legend1', 2);
  createHistogramLegend('histogram-legend0', 0);
  createHistogramLegend('histogram-legend1', 1);

  createHistogramLegend('histogram-legend0_nocurves', 0);
  createHistogramLegend('histogram-legend1_nocurves', 1);


  
  

  // Create outcome and profit curves
  displayCurves('single-outcomes', singleModel_interactive, tprOutcome / singleModel_interactive.items.length, fprOutcome / singleModel_interactive.items.length)
  displayCurves('single-profits', singleModel_interactive, tprValue, fprValue)

  displayCurves('outcomes0', comparisonExample0, tprOutcome / comparisonExample0.items.length, fprOutcome / comparisonExample0.items.length)
  displayCurves('profits0', comparisonExample0, tprValue, fprValue)

  displayCurves('outcomes1', comparisonExample1, tprOutcome / comparisonExample1.items.length, fprOutcome / comparisonExample1.items.length)
  displayCurves('profits1', comparisonExample1, tprValue, fprValue)
  
  function updateTextDisplays(event) {
    // Update number readouts.
    function display(id, value) {
      var element = document.getElementById(id);
      element.innerHTML = '' + value;
      element.style.color = value < 0 ? '#f00' : '#000';
    }
    display('single-profit0', singleModel_nocurves.profit);
    display('single-profit1', singleModel_interactive.profit);
    display('single-scorechange0', singleModel_nocurves.chg);
    display('single-scorechange1', singleModel_interactive.chg);

    display('comparison0_chg', comparisonExample0_nocurves.chg);
    display('comparison1_chg', comparisonExample1_nocurves.chg);


    display('total-profit', comparisonExample0.profit +
        comparisonExample1.profit);
    display('total-profit_nocurves', comparisonExample0_nocurves.profit +
        comparisonExample1_nocurves.profit);

    // Update micro-story annotations.
    function annotate(id) {
      var annotations = document.getElementsByClassName(id + '-annotation');
      for (var i = 0; i < annotations.length; i++) {
        annotations[i].style.visibility = id == event ? 'visible' : 'hidden';
        annotations[i].style.display = id == event ? 'block' : 'none';
      }
    }
    // Annotate each of the criteria defined by our optimizer.
    //annotate(MAX_PROFIT);
    // annotate(GROUP_UNAWARE);
    // annotate(DEMOGRAPHIC_PARITY);
    // annotate(EQUAL_OPPORTUNITY);
  }

  // Update text whenever any of the interactive models change.
  singleModel_interactive.addListener(updateTextDisplays);
  singleModel_nocurves.addListener(updateTextDisplays);
  comparisonExample0.addListener(updateTextDisplays);
  comparisonExample1.addListener(updateTextDisplays);
  comparisonExample0_nocurves.addListener(updateTextDisplays);
  comparisonExample1_nocurves.addListener(updateTextDisplays);



    // Initialize everything.
  comparisonExample0.classify(50);
  comparisonExample1.classify(50);  
  comparisonExample0_nocurves.classify(50);
  comparisonExample1_nocurves.classify(50);
  singleModel_interactive.classify(50);
  singleModel_nocurves.classify(50);

  singleModel_interactive.notifyListeners();
  singleModel_nocurves.notifyListeners();
  singleModel.notifyListeners();
  comparisonExample0.notifyListeners();
  comparisonExample1.notifyListeners();
  comparisonExample0_nocurves.notifyListeners();
  comparisonExample1_nocurves.notifyListeners();
}




// Models for threshold classifiers
// along with simple optimization code.

// An item with an intrinsic value, predicted classification, and
// a "score" for use by a threshold classifier.
// The going assumption is that the values and predicted values
// are 0 or 1. Furthermore "1" is considered a positive/good value.
var Item = function(category, value, score) {
  this.category = category;
  this.value = value;
  this.predicted = value;
  this.score = score;
};


// A group model defines a group of items, with a threshold
// for a classifier and associated values for true and
// false positives. It also can notify listeners that an event
// has occurred to change the model.
var GroupModel = function(items, tprValue, fprValue, tprOutcome=0, fprOutcome=0) {
  // Data defining the model.
  this.items = items;
  this.tprValue = tprValue;
  this.fprValue = fprValue;
  this.tprOutcome = tprOutcome;
  this.fprOutcome = fprOutcome;
  // Observers of the model; needed for interactive diagrams.
  this.listeners = [];
};

// Classify according to the given threshold, and store various
// interesting metrics for future use.
GroupModel.prototype.classify = function(threshold, newTpr = false, newFpr = false) {
  this.threshold = threshold;

  var numEqual = 0;
  // Classify and find positive rates.
  this.items.forEach(function(item) {
    if (item.score > threshold){
      item.predicted = 1;
    } else if (item.score < threshold) {
      item.predicted = 0;
    } else if (item.score == threshold) {
      item.predicted = 1;
      numEqual += 1;
    }
  });
  // this.items.forEach(function(item) {
  //   if (item.score == threshold){

  //   }
  // });
  this.tpr = tpr(this.items);
  this.positiveRate = positiveRate(this.items);

  // Find profit.
  this.profit = profit(this.items, this.tprValue, this.fprValue);
  this.chg = Math.round(profit(this.items, this.tprOutcome, this.fprOutcome) / this.items.length);
  if (newTpr && newFpr) {
    return profit(this.items, newTpr, newFpr);
  }
};

GroupModel.prototype.setMaps = function(stepSize = 1) {
  var tpr_list = []; var accrate_list = [];
  var threshold_list = []; 
  for (var t = 0; t <= 100; t += stepSize) {
    this.classify(t);
    tpr_list.push(this.tpr)
    accrate_list.push(this.positiveRate)
    threshold_list.push(t)
  }
  this.maps = [threshold_list, accrate_list, tpr_list]
}

// GroupModels follow a very simple observer pattern; they
// have listeners which can be notified of arbitrary events.
GroupModel.prototype.addListener = function(listener) {
  this.listeners.push(listener);
};

// Tell all listeners of the specified event.
GroupModel.prototype.notifyListeners = function(event) {
  this.listeners.forEach(function(listener) {listener(event);});
};

// Create items whose scores have a
// "deterministic normal" distribution. That is, the items track
// a Gaussian curve. This not the same as actually choosing scores
// normally, but for expository purposes it's useful to have
// deterministic, smooth distributions of values.
function makeNormalItems(category, value, n, mean, std) {
  var items = [];
  var error = 0;
  for (var score = 0; score < 100; score++) {
    var e = error + n * Math.exp(-(score - mean) * (score - mean) / (2 * std * std)) /
            (std * Math.sqrt(2 * Math.PI));
    var m = Math.floor(e);
    error = e - m;
    for (var j = 0; j < m; j++) {
      items.push(new Item(category, value, score));
    }
  }
  return items;
}

function getNormalPis(mean, std) {
  var pi = [];
  var repay_prob = [];
  for (var score = 0; score <= 100; score++) {
    pi[score] = Math.exp(-(score - mean) * (score - mean) / (2 * std * std)) /
            (std * Math.sqrt(2 * Math.PI));
    repay_prob[score] = 1 / (1 + Math.exp(-0.1*(score-40)))
    //repay_prob[score] = (score / 100)**2
  }
  return [pi, repay_prob]
}


function getFICOEmpPis() {
  var piA = [ 0, 0.0119,  0.0528,  0.028 ,  0.0355,  0.0234,  0.0465,  0.0312,
        0.0258,  0.0204,  0.0183,  0.0178,  0.0217,  0.0274,  0.0214,
        0.0254,  0.0216,  0.0197,  0.017 ,  0.0149,  0.0122,  0.0475,
        0.0186,  0.0147,  0.0149,  0.0176,  0.0146,  0.0146,  0.0146,
        0.0141,  0.0124,  0.012 ,  0.0113,  0.0108,  0.0108,  0.0101,
        0.0105,  0.0091,  0.0105,  0.0103,  0.0076,  0.008 ,  0.0074,
        0.0094,  0.0079,  0.0089,  0.0074,  0.0057,  0.0066,  0.0058,
        0.0054,  0.0055,  0.0049,  0.0045,  0.0058,  0.0043,  0.0058,
        0.0044,  0.0033,  0.0036,  0.0044,  0.0043,  0.0039,  0.0039,
        0.0042,  0.0029,  0.0026,  0.0025,  0.0034,  0.0017,  0.0027,
        0.0028,  0.002 ,  0.0044,  0.0022,  0.004 ,  0.0022,  0.0044,
        0.0027,  0.0021,  0.0037,  0.0028,  0.0021,  0.0025,  0.0025,
        0.0032,  0.0022,  0.0025,  0.0016,  0.0019,  0.0011,  0.0009,
        0.0029,  0.0024,  0.0024,  0.0017,  0.0013,  0.0038,  0.001 ,
        0.0001];
  var piB = [ 0, 0.0026,  0.0117,  0.0066,  0.0087,  0.0058,  0.0118,  0.0084,
        0.0078,  0.0068,  0.0058,  0.006 ,  0.0074,  0.0098,  0.0084,
        0.0099,  0.0086,  0.0082,  0.007 ,  0.0071,  0.0066,  0.0138,
        0.0076,  0.0074,  0.0075,  0.0096,  0.0075,  0.0083,  0.0095,
        0.0093,  0.0116,  0.0088,  0.0099,  0.009 ,  0.0092,  0.0103,
        0.0099,  0.0108,  0.0111,  0.0105,  0.009 ,  0.0092,  0.0085,
        0.0102,  0.0101,  0.0118,  0.0102,  0.0094,  0.0109,  0.0118,
        0.0095,  0.0091,  0.0099,  0.0098,  0.0106,  0.0093,  0.0103,
        0.0096,  0.0097,  0.0104,  0.0106,  0.0099,  0.0101,  0.0104,
        0.0113,  0.0089,  0.0095,  0.0093,  0.0102,  0.0054,  0.0115,
        0.0085,  0.0067,  0.016 ,  0.0081,  0.0145,  0.009 ,  0.0165,
        0.0147,  0.0108,  0.0117,  0.0155,  0.0099,  0.0125,  0.0132,
        0.0145,  0.0124,  0.0122,  0.0121,  0.0137,  0.0096,  0.0077,
        0.0169,  0.0155,  0.0205,  0.0128,  0.0092,  0.0217,  0.0097,
        0.0009];
  var repay_A = [ 0, 0.011 ,  0.0284,  0.0458,  0.0632,  0.0807,  0.0981,  0.1155,
        0.1329,  0.1503,  0.1678,  0.1833,  0.1933,  0.2034,  0.2135,
        0.2235,  0.2335,  0.2435,  0.2535,  0.2637,  0.2737,  0.2989,
        0.3699,  0.4409,  0.5119,  0.5828,  0.6538,  0.7248,  0.7957,
        0.8667,  0.9377,  1.0018,  1.045 ,  1.0884,  1.1316,  1.1749,
        1.2182,  1.2615,  1.3048,  1.348 ,  1.3914,  1.4308,  1.459 ,
        1.4872,  1.5153,  1.5435,  1.5716,  1.5998,  1.6279,  1.6561,
        1.6842,  1.7067,  1.7121,  1.7175,  1.7229,  1.7283,  1.7338,
        1.7392,  1.7446,  1.75  ,  1.7554,  1.7616,  1.7703,  1.779 ,
        1.7877,  1.7964,  1.8051,  1.8138,  1.8224,  1.8312,  1.8398,
        1.8481,  1.855 ,  1.8637,  1.8723,  1.8793,  1.8862,  1.8931,
        1.9035,  1.9104,  1.9151,  1.9131,  1.9111,  1.9091,  1.9071,
        1.9051,  1.9031,  1.9011,  1.8991,  1.8969,  1.8976,  1.9061,
        1.9168,  1.9275,  1.9359,  1.9445,  1.953 ,  1.9615,  1.9701,
        1.9786];
  var repay_B = [ 0.0351,  0.0587,  0.0824,  0.1061,  0.1297,  0.1533,  0.1769,
        0.2005,  0.2241,  0.2477,  0.2722,  0.2994,  0.3266,  0.3538,
        0.3808,  0.408 ,  0.4352,  0.4624,  0.4896,  0.5168,  0.5559,
        0.6305,  0.7051,  0.7797,  0.8543,  0.9289,  1.0035,  1.0781,
        1.1527,  1.2273,  1.2925,  1.3294,  1.3662,  1.403 ,  1.4398,
        1.4766,  1.5135,  1.5504,  1.5872,  1.624 ,  1.6562,  1.6744,
        1.6927,  1.7109,  1.7292,  1.7474,  1.7657,  1.7839,  1.8022,
        1.8204,  1.8359,  1.8432,  1.8504,  1.8576,  1.8649,  1.8722,
        1.8794,  1.8866,  1.8938,  1.9012,  1.9075,  1.9112,  1.9149,
        1.9185,  1.9223,  1.9259,  1.9297,  1.9333,  1.937 ,  1.9407,
        1.944 ,  1.9463,  1.9491,  1.952 ,  1.9542,  1.9564,  1.9587,
        1.9621,  1.9644,  1.9661,  1.9666,  1.9671,  1.9675,  1.9679,
        1.9685,  1.9689,  1.9693,  1.9698,  1.9703,  1.9709,  1.9721,
        1.9735,  1.9749,  1.9761,  1.9772,  1.9783,  1.9795,  1.9807,
        1.9817];
  return [piA, piB, repay_A, repay_B]
}

function getFICOPis() {
  var piA = [ 0.06071798,  0.04208177,  0.03541181,  0.0315487 ,  0.02888477,
        0.02687177,  0.02526226,  0.02392507,  0.0227829 ,  0.02178666,
        0.02090335,  0.02010982,  0.01938922,  0.01872894,  0.01811934,
        0.01755285,  0.01702346,  0.0165263 ,  0.0160574 ,  0.01561346,
        0.0151917 ,  0.01478981,  0.01440578,  0.0140379 ,  0.0136847 ,
        0.0133449 ,  0.01301735,  0.01270108,  0.0123952 ,  0.01209893,
        0.01181158,  0.01153253,  0.0112612 ,  0.01099711,  0.01073978,
        0.0104888 ,  0.0102438 ,  0.01000442,  0.00977035,  0.00954131,
        0.00931701,  0.00909721,  0.0088817 ,  0.00867024,  0.00846266,
        0.00825877,  0.00805839,  0.00786137,  0.00766757,  0.00747684,
        0.00728905,  0.00710409,  0.00692185,  0.0067422 ,  0.00656506,
        0.00639034,  0.00621793,  0.00604775,  0.00587973,  0.00571379,
        0.00554986,  0.00538787,  0.00522775,  0.00506945,  0.00491291,
        0.00475807,  0.00460487,  0.00445327,  0.00430322,  0.00415468,
        0.0040076 ,  0.00386193,  0.00371765,  0.0035747 ,  0.00343306,
        0.00329269,  0.00315355,  0.00301561,  0.00287884,  0.00274322,
        0.00260871,  0.00247528,  0.00234291,  0.00221158,  0.00208126,
        0.00195192,  0.00182354,  0.0016961 ,  0.00156958,  0.00144396,
        0.00131922,  0.00119533,  0.00107229,  0.00095007,  0.00082866,
        0.00070804,  0.00058819,  0.0004691 ,  0.00035074,  0.00023312];
  var piB = [ 0.00148771,  0.00222933,  0.00278385,  0.00324451,  0.00364632,
        0.00400678,  0.00433611,  0.0046409 ,  0.00492566,  0.00519366,
        0.00544736,  0.00568864,  0.00591902,  0.0061397 ,  0.00635167,
        0.00655577,  0.00675269,  0.00694304,  0.00712731,  0.00730596,
        0.00747938,  0.0076479 ,  0.00781183,  0.00797143,  0.00812694,
        0.00827859,  0.00842656,  0.00857102,  0.00871214,  0.00885006,
        0.00898491,  0.0091168 ,  0.00924585,  0.00937216,  0.00949581,
        0.00961689,  0.00973548,  0.00985165,  0.00996546,  0.01007697,
        0.01018623,  0.0102933 ,  0.01039822,  0.01050102,  0.01060175,
        0.01070044,  0.01079712,  0.01089181,  0.01098454,  0.01107533,
        0.01116419,  0.01125114,  0.01133619,  0.01141935,  0.01150061,
        0.01157999,  0.01165748,  0.01173307,  0.01180676,  0.01187853,
        0.01194838,  0.01201628,  0.0120822 ,  0.01214613,  0.01220802,
        0.01226786,  0.01232558,  0.01238114,  0.0124345 ,  0.01248559,
        0.01253434,  0.01258067,  0.01262451,  0.01266574,  0.01270427,
        0.01273996,  0.01277269,  0.01280228,  0.01282857,  0.01285135,
        0.01287039,  0.0128854 ,  0.0128961 ,  0.01290211,  0.01290301,
        0.01289829,  0.01288737,  0.01286951,  0.01284383,  0.01280924,
        0.01276437,  0.01270743,  0.01263609,  0.01254718,  0.01243626,
        0.01229677,  0.01211845,  0.01188382,  0.01155915,  0.01106431];
  var repay_prob = [];
  for (var score = 0; score <= 100; score++) {
    repay_prob[score] = 1 / (1 + Math.exp(-0.25*(score-30))) // (score / 100)
  }
  return [piA, piB, repay_prob]
}

function makeItems(category, pi, repay_prob, total_items) {
  var items_repay = []; var items_default = [];
  var error_repay = 0; var error_default = 0;
  var num_repay_by_score = [];
  var num_default_by_score = [];
  for (var score = 0; score < 100; score++) {
    var num_repay = error_repay + pi[score] * repay_prob[score] * total_items;
    var num_repay_int = Math.floor(num_repay);
    error_repay = num_repay - num_repay_int;
    num_repay_by_score.push(num_repay_int)
    for (var j = 0; j < num_repay_int; j++) {
      items_repay.push(new Item(category, 1, score));
    }

    var num_default = error_default + pi[score] * (1 - repay_prob[score]) * total_items;
    var num_default_int = Math.floor(num_default);
    error_default = num_default - num_default_int;
    num_default_by_score.push(num_default_int)
    for (var j = 0; j < num_default_int; j++) {
      items_default.push(new Item(category, 0, score));
    }
  }
  console.log(num_repay_by_score)
  console.log(num_default_by_score)
  return items_repay.concat(items_default);
}

function getHardcodedSingleModel(category) {
  var num_repay_by_score =   [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1,1,1,1,2,2,2,2,3,2,3,3,4,4,4,4,5,5,5,6,6,6,7,7,6,8,7,7,8,7,8,8,7,8,7,7,7,7,7,7,6,6,6,6,5,5,5,5,4,4,4,3,4,3,2,3,2,3,2,2,1,2,1,2,1,1,1,0,1,1,0,1,0,1,0,0,1]
  var num_default_by_score = [0,1,1,1,2,1,2,1,2,2,2,2,3,2,4,3,3,4,4,4,4,4,5,5,5,5,5,6,5,6,6,6,5,6,6,6,6,5,6,5,5,5,4,5,4,4,4,3,3,3,3,3,2,2,2,1,2,1,2,1,1,0,1,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
  var items_repay = []; var items_default = [];
  for (var score = 0; score < 100; score++) {
    for (var j = 0; j < num_repay_by_score[score]; j++) {
      items_repay.push(new Item(category, 1, score));
    }
    for (var j = 0; j < num_default_by_score[score]; j++) {
      items_default.push(new Item(category, 0, score));
    }
  }
  return items_repay.concat(items_default);
}



// Profit of a model, subject to the given values
// for true and false positives. Note that the simple model
// in the paper assumes zero value for negatives.
function profit(items, tprValue, fprValue) {
  var sum = 0;
  items.forEach(function(item) {
    if (item.predicted == 1) {
      sum += item.value == 1 ? tprValue : fprValue;
    }
  });
  return sum;
}


// Count specified type of items.
function countMatches(items, value, predicted) {
  var n = 0;
  items.forEach(function(item) {
    if (item.value == value && item.predicted == predicted) {
      n++;
    }
  });
  return n;
}

// Calculate true positive rate
function tpr(items) {
  var totalGood = 0;
  var totalGoodPredictedGood = 0;
  items.forEach(function(item) {
    totalGood += item.value;
    totalGoodPredictedGood += item.value * item.predicted;
  });
  if (totalGood == 0) {
    return 1;
  }
  return totalGoodPredictedGood / totalGood;
}

// Calculate overall positive rate
function positiveRate(items) {
  var totalGood = 0;
  items.forEach(function(item) {
    totalGood += item.predicted;
  });
  return totalGood / items.length;
}

// Given our set-up, we can't always hope for exact equality
// of various ratios.
// We test for two numbers to be close enough that they look
// the same when formatted for display.
// This is not technically optimal mathematically but definitely
// optimal pedagogically!
function approximatelyEqual(x, y, scale = 100) {
  return Math.round(scale * x) == Math.round(scale * y);
}

function approximatelyEqualAcc(x, y, tol = 1e-1) {
  return Math.abs(x-y) < tol; // Math.round(scale * x) == Math.round(scale * y);
}

// Constants for types of optimization.
var MAX_PROFIT = 'max-profit';
var GROUP_UNAWARE = 'group-unaware';
var DEMOGRAPHIC_PARITY = 'demographic-parity';
var EQUAL_OPPORTUNITY = 'equal-opportunity';


// Returns an object with four functions representing the four
// ways to optimize between two models that are described
// in the blog post.
function Optimizer(model0, model1, stepSize) {
  function classify(t0, t1) {
    model0.classify(t0);
    model1.classify(t1);
    return model0.profit + model1.profit;
  }

  // Get extents of item scores, and use for range of search.
  function getScore(item) {return item.score;}
  var extent0 = d3.extent(model0.items, getScore);
  var extent1 = d3.extent(model1.items, getScore);
  // Add to max value to include possibility of all-negative threshold.
  extent0[1] += stepSize;
  extent1[1] += stepSize;

  // Maximize utility according to the given constraint.
  // The constraint function takes the two thresholds as arguments.
  // Although an exhautive search works fine here, note that there
  // is a huge amount of room for optimization. See paper by Hardt et al.
  // for additional algorithmic discussion.
  function maximizeWithConstraint(constraint, event) {
    var maxProfit = -Infinity;
    var bestT0;
    var bestT1;
    for (var t0 = extent0[0]; t0 <= extent0[1]; t0 += stepSize) {
      for (var t1 = extent1[0]; t1 <= extent1[1]; t1 += stepSize) {
        var p = classify(t0, t1);
        if (!constraint(t0, t1)) {continue;}
        if (p > maxProfit) {
          maxProfit = p;
          bestT0 = t0;
          bestT1 = t1;
        }
      }
    }
    classify(bestT0, bestT1);
    model0.notifyListeners(event);
    model1.notifyListeners(event);
  }

  // Return a bundle of optimizer functions,
  return {
    // Maximize utility, allowing any combination of thresholds.
    maximizeProfit: function() {
      model0.link_setting = 'max-profit'
      model1.link_setting = 'max-profit'
      setButtonColor('max-profit');
      maximizeWithConstraint(function() {return true;}, MAX_PROFIT);
    },
    // Group unware: thresholds must be equal in both groups.
    groupUnaware: function() {
      maximizeWithConstraint(function(t0, t1) {
        return t0 == t1;
      }, GROUP_UNAWARE);
    },
    // Demographic parity: true + false positive rates must be the same.
    demographicParity: function() {
      model0.link_setting = 'demographic-parity'
      model1.link_setting = 'demographic-parity'
      setButtonColor('demographic-parity');
      maximizeWithConstraint(function(t0, t1) {
        var pr0 = positiveRate(model0.items);
        var pr1 = positiveRate(model1.items);
        return approximatelyEqual(pr0, pr1);
      }, DEMOGRAPHIC_PARITY);
    },
    // Equal opportunity: true positive rates must be the same.
    equalOpportunity: function() {
      model0.link_setting = 'equal-opportunity'
      model1.link_setting = 'equal-opportunity'
      setButtonColor('equal-opportunity');
      maximizeWithConstraint(function(t0, t1) {
        var tpr0 = tpr(model0.items);
        var tpr1 = tpr(model1.items);
        return approximatelyEqual(tpr0, tpr1);
      }, EQUAL_OPPORTUNITY);
    }
  };
}

function setButtonColor(button_name) {
  switch(button_name) {
    case 'max-profit':
        document.getElementById('max-profit').style.opacity = 1;
        document.getElementById('demographic-parity').style.opacity = 0.5;
        document.getElementById('equal-opportunity').style.opacity = 0.5;
        break;
    case 'demographic-parity':
        document.getElementById('max-profit').style.opacity = 0.5;
        document.getElementById('demographic-parity').style.opacity = 1;
        document.getElementById('equal-opportunity').style.opacity = 0.5;
        break;
    case 'equal-opportunity':
        document.getElementById('max-profit').style.opacity = 0.5;
        document.getElementById('demographic-parity').style.opacity = 0.5;
        document.getElementById('equal-opportunity').style.opacity = 1;
        break;
  } 
  //'max-profit'
}

// Interactive diagrams for fairness demo.
// These are lightweight components customized
// for this demo.

// Side of grid in histograms and correctness matrices.
var SIDE = 7;
var Y_SIDE_RATIO = 1;
// Component dimensions.
var HEIGHT = 250;
var HISTOGRAM_WIDTH = 370;
var HISTOGRAM_LEGEND_HEIGHT = 60;

// Histogram bucket width
var HISTOGRAM_BUCKET_SIZE = 2;

// Padding on left; needed within SVG so annotations show up.
var LEFT_PAD = 10;

// Palette constants and functions.

// Colors of categories of items.
var CATEGORY_COLORS = ['#039', '#dc7633 ', '#039'];

// Colors for pie slices; set by hand because of various tradeoffs.
// Order:  false negative, true negative, true positive, false positive.
var PIE_COLORS = [['#686868', '#ccc','#039', '#92a5ce'],
                  ['#686868', '#ccc','#c70',  '#f0d6b3']];

function itemColor(category, predicted) {
  return predicted == 0 ? '#555' : CATEGORY_COLORS[category];
}

function itemOpacity(value) {
  return .3 + .7 * value;
}

function iconColor(d) {
  return d.predicted == 0 && !d.colored ? '#555' : CATEGORY_COLORS[d.category];
}

function iconOpacity(d) {
  return itemOpacity(d.value);
}

// Icon for a person in histogram or correctness matrix.
function defineIcon(selection) {
  selection
    .attr('class', 'icon')
    .attr('stroke', iconColor)
    .attr('fill', iconColor)
    .attr('fill-opacity', iconOpacity)
    .attr('stroke-opacity', function(d) {return .4 + .6 * d.value;})
    .attr('cx', function(d) {return d.x + d.side / 2;})
    .attr('cy', function(d) {return d.y + d.side_y / 2;})
    .attr('r', function(d) {return d.side_y * .4});
}

function createIcons(id, items, width, height, pad) {
  var svg = d3.select('#' + id).append('svg')
    .attr('width', width)
    .attr('height', height);
  if (pad) {
    svg = svg.append('g').attr('transform', 'translate(' + pad + ',0)');
  }
  var icon = svg.selectAll('.icon')
    .data(items)
  .enter().append('circle')
    .call(defineIcon);
  return svg;
}

function gridLayout(items, x, y, ncols = 15) {
  items = items.reverse();
  var n = items.length;
  var cols = ncols;
  var rows = Math.ceil(n / cols);
  items.forEach(function(item, i) {
    item.x = x + SIDE * (i % cols);
    item.y = y + 0.5 * SIDE * Y_SIDE_RATIO * Math.floor(i / cols);
    item.side = SIDE;
    item.side_y = 0.5 * SIDE * Y_SIDE_RATIO;
  });
}


// Shallow copy of item array.
function copyItems(items) {
  return items.map(function(item) {
    var copy = new Item(item.category, item.value, item.score);
    copy.predicted = item.predicted;
    return copy;
  });
}

// Create histogram for scores of items in a model.
function createHistogram(id, model, noThreshold, includeAnnotation, linked_model) {
  var width = HISTOGRAM_WIDTH;
  var height = HEIGHT;
  var bottom = height - 16;

  // Create an internal copy.
  var items = copyItems(model.items);

  // Icons
  var numBuckets = 100 / HISTOGRAM_BUCKET_SIZE;
  var pedestalWidth = numBuckets * SIDE;
  var hx = (width - pedestalWidth) / 2;
    var scale = d3.scaleLinear().range([hx, hx + pedestalWidth]).
      domain([0, 100]);
  var scale2 = d3.scaleLinear().range([hx, hx + pedestalWidth]).
      domain([300, 800]);

  function histogramLayout(items, x, y, side, low, high, bucketSize, y_side_ratio) {
    var buckets = [];
    var maxNum = Math.floor((high - low) / bucketSize);
    items.forEach(function(item) {
      var bn = Math.floor((item.score - low) / bucketSize);
      bn = Math.max(0, Math.min(maxNum, bn));
      buckets[bn] = 1 + (buckets[bn] || 0);
      item.x = x + side * bn;
      item.y = y - side * y_side_ratio * buckets[bn];
      item.side = side;
      item.side_y = side * y_side_ratio;
    });
  }

  histogramLayout(items, hx, bottom, SIDE, 0, 100, HISTOGRAM_BUCKET_SIZE, Y_SIDE_RATIO);
  var svg = createIcons(id, items, width, height);

  var tx = width / 2;
  var topY = 60;
  var axis = d3.axisBottom(scale2);
  svg.append('g').attr('class', 'histogram-axis')
    .attr('transform', 'translate(0,-8)')
    .call(axis);
  d3.select('.domain').attr('stroke-width', 1);

  if (noThreshold) {
    return;
  }
  // Sliding threshold bar.
  var cutoff = svg.append('rect').attr('x', tx - 2).attr('y', topY - 10).
      attr('width', 3).attr('height', height - topY);

  var thresholdLabel = svg.append('text').text('loan threshold: 550') // 50 * (500/100) + 300
      .attr('x', tx)
      .attr('y', 40)
      .attr('text-anchor', 'middle').attr('class', 'bold-label');

  if (includeAnnotation) {
    var annotationPad = 10;
    var annotationW = 200;
    var thresholdAnnotation = svg.append('rect')
        .attr('class', 'annotation group-unaware-annotation')
        .attr('x', tx - annotationW / 2)
        .attr('y', 30 - annotationPad)
        .attr('rx', 20)
        .attr('ry', 20)
        .attr('width', annotationW)
        .attr('height', 30);
   }

  function setThreshold(t, eventFromUser) {
    t = Math.max(0, Math.min(t, 100));
    if (eventFromUser) {
      t = HISTOGRAM_BUCKET_SIZE * Math.round(t / HISTOGRAM_BUCKET_SIZE);
    } else {
      tx = Math.round(scale(t));
      t = HISTOGRAM_BUCKET_SIZE * Math.round(t / HISTOGRAM_BUCKET_SIZE)
    }
    tx = Math.max(0, Math.min(width - 4, tx));
    var rounded = SIDE * Math.round(tx / SIDE);
    cutoff.attr('x', rounded);
    var labelX = Math.max(50, Math.min(rounded, width - 70));
    thresholdLabel.attr('x', labelX).text('loan threshold: ' + (t * 5 + 300));
    if (includeAnnotation) {
      thresholdAnnotation.attr('x', tx - annotationW / 2);
    }
    svg.selectAll('.icon').call(defineIcon);
  }
  var drag = d3.drag()
    .on('drag', function() {
      var oldTx = tx;
      tx += d3.event.dx;
      var t = scale.invert(tx);
      setThreshold(t, true);
      if (tx != oldTx) {
        model.classify(t);
        model.notifyListeners('histogram-drag');
      }
  });
  svg.call(drag);
  model.addListener(function(event) {
    for (var i = 0; i < items.length; i++) {
      items[i].predicted = items[i].score >= model.threshold ? 1 : 0;
    }
    if (!(typeof linked_model == "undefined") && event == 'histogram-drag') {
      translateThresh(model, linked_model, model.link_setting) 
      //linked_model.classify(model.threshold);
      //linked_model.notifyListeners('linked-model')// update other model with model.threshold
    }
    setThreshold(model.threshold, event == 'histogram-drag');
  });
}

function getTranslatedThresh(linked_model, match_list, match) {
  [threshold_list, accrate_list, tpr_list] = linked_model.maps;
  var translated_threshold = linked_model.threshold;
  var translated_index = 0;

  // for (var i = 0; i < threshold_list.length; i += 1) {
  for (var i = threshold_list.length-1; i >= 0; i -= 1) {
    // if (approximatelyEqual(1, match, 1000)) {
    //   break;
    // }
    if (approximatelyEqual(match_list[i], match, 100)) { 
      translated_threshold = threshold_list[i]; 
      translated_index = i;
      break;
    }
  }
  // var translated_index = threshold_list.length-1;
  return [translated_threshold, translated_index]
}

function translateThresh(model, linked_model, link_type) {
  var match_list; var match;
  if (link_type == "demographic-parity") {
    match_list = linked_model.maps[1]; // accrate_list;
    match = model.positiveRate;
  } else if (link_type == "equal-opportunity") {
      match_list = linked_model.maps[2];
      match = model.tpr;
  } else {
      return;
  }
  [translated_threshold, idx] = getTranslatedThresh(linked_model, match_list, match)
  linked_model.classify(translated_threshold);
  linked_model.notifyListeners('linked-model')// update other model with model.threshold
}

// Draw full legend for histogram, with all four possible
// categories of people.
function createHistogramLegend(id, category) {
  var width = HISTOGRAM_WIDTH;
  var height = HISTOGRAM_LEGEND_HEIGHT;
  var centerX = width / 2;
  var boxSide = 16;
  var centerPad = 1;

  // Create SVG.
  var svg = d3.select('#' + id).append('svg')
    .attr('width', width)
    .attr('height', height);

  // Create boxes.
  svg.append('rect').attr('width', boxSide).attr('height', boxSide)
    .attr('x', centerX - boxSide - centerPad).attr('y', boxSide)
    .attr('fill', itemColor(category, 0))
    .attr('fill-opacity', itemOpacity(1));
  svg.append('rect').attr('width', boxSide).attr('height', boxSide)
    .attr('x', centerX + centerPad).attr('y', boxSide)
    .attr('fill', itemColor(category, 1))
    .attr('fill-opacity', itemOpacity(1));

  svg.append('rect').attr('width', boxSide).attr('height', boxSide)
    .attr('x', centerX - boxSide - centerPad).attr('y', 0)
    .attr('fill', itemColor(category, 0))
    .attr('fill-opacity', itemOpacity(0));
  svg.append('rect').attr('width', boxSide).attr('height', boxSide)
    .attr('x', centerX + centerPad).attr('y', 0)
    .attr('fill', itemColor(category, 1))
    .attr('fill-opacity', itemOpacity(0));

  // Draw text.
  var textPad = 4;
  svg.append('text')
      .text('denied loan / would pay back')
      .attr('x', centerX - boxSide - textPad)
      .attr('y', 2 * boxSide - textPad)
      .attr('text-anchor', 'end')
      .attr('class', 'legend-label');
  svg.append('text')
      .text('denied loan / would default')
      .attr('x', centerX - boxSide - textPad)
      .attr('y', boxSide - textPad)
      .attr('text-anchor', 'end')
      .attr('class', 'legend-label');
  svg.append('text')
      .text('granted loan / pays back')
      .attr('x', centerX + boxSide + textPad)
      .attr('y', 2 * boxSide - textPad)
      .attr('text-anchor', 'start')
      .attr('class', 'legend-label');
  svg.append('text')
      .text('granted loan / defaults')
      .attr('x', centerX + boxSide + textPad)
      .attr('y', boxSide - textPad)
      .attr('text-anchor', 'start')
      .attr('class', 'legend-label');
}

// A much simpler legend, used in the top diagram,
// with only two categories of people and a different layout.
function createSimpleHistogramLegend(id, category) {
  var width = HISTOGRAM_WIDTH;
  var height = HISTOGRAM_LEGEND_HEIGHT;
  var centerX = width / 2;
  var boxSide = 16;
  var centerPad = 1;
  var lx = 50;

  // Create SVG.
  var svg = d3.select('#' + id).append('svg')
    .attr('width', width)
    .attr('height', height);

  // Create boxes.
  svg.append('rect').attr('width', boxSide).attr('height', boxSide)
    .attr('x', centerX + centerPad).attr('y', 0)
    .attr('fill', itemColor(category, 1))
    .attr('fill-opacity', itemOpacity(1));
  svg.append('rect').attr('width', boxSide).attr('height', boxSide)
    .attr('x', lx).attr('y', 0)
    .attr('fill', itemColor(category, 1))
    .attr('fill-opacity', itemOpacity(0));

  // Draw text.
  var textPad = 4;
  svg.append('text')
      .text('would pay back loan')
      .attr('x', centerX + boxSide + textPad)
      .attr('y', boxSide - textPad)
      .attr('text-anchor', 'start')
      .attr('class', 'legend-label');
  svg.append('text')
      .text('would default on loan')
      .attr('x', lx + boxSide + textPad)
      .attr('y', boxSide - textPad)
      .attr('text-anchor', 'start')
      .attr('class', 'legend-label');
}

// Create a nice label for percentages; the return value is a callback
// to update the number.
function createPercentLabel(svg, x, y, text, labelClass, statClass) {
  var label = svg.append('text').text(text)
      .attr('x', x).attr('y', y).attr('class', labelClass);
  var labelWidth = label.node().getComputedTextLength();
  var stat = svg.append('text').text('')
      .attr('x', x + labelWidth + 4).attr('y', y).attr('class', statClass);

  // Return a function that updated the label.
  return function(value) {
    var formattedValue = Math.round(100 * value) + '%';
    stat.text(formattedValue);
  }
}

// Helper for multiline explanations.
function explanation(svg, lines, x, y) {
  lines.forEach(function(line) {
    svg.append('text').text(line)
        .attr('x', x).attr('y', y += 16).attr('class', 'explanation');
  });
}

// Create two pie charts: 1. for all classification rates
// and 2. true positive rates.

// Creates matrix view of dots representing correct and
// incorrect items.

// Creates matrix view of dots representing no loans
function createLoanMatrix(id, model) {
  var width = 300;
  var noloanY = 180; var topY = 18;
  var height = 1.7*noloanY;
  var noloan, truepositive, falsepositive;
  function layout() {
    truepositive = model.items.filter(function(item) {
      return item.predicted == 1 && item.value == item.predicted;
    });
    falsepositive = model.items.filter(function(item) {
      return item.predicted == 1 && item.value != item.predicted;
    });
    noloan = model.items.filter(function(item) {
      return item.predicted == 0;
    });
    gridLayout(truepositive, 2, 80, 20);
    gridLayout(falsepositive, width / 2 + 4, 80, 10);
    gridLayout(noloan, 2, 80 - topY + noloanY, 30);
  }

  layout();
  var svg = createIcons(id, model.items, width, height, LEFT_PAD);

  
  var offset = 20;
  var selRateLabel = createPercentLabel(svg, 0, topY, 'Selection Rate',
      'pie-label1', 'pie-number');
  var truePosLabel = createPercentLabel(svg, 0, topY+offset, 'Correct',
      'pie-label', 'pie-number');
  var falsePosLabel = createPercentLabel(svg, width / 2 + 4, topY+offset, 'Incorrect',
      'pie-label', 'pie-number');


  var noloanLabel = createPercentLabel(svg, 0, noloanY, 'Denied Loans',
      'pie-label', 'pie-number');


  // Add explanation of correct decisions.
  explanation(svg, ['repaid loans result in',
      'increased credit score'], 0, topY+offset);
  explanation(svg, ['defaults resut in', 'lowered score'], width / 2 + 4, topY+offset);

  explanation(svg, ['denied applicants experience no', 'credit score change and contribute',
    'nothing to profit'], 
    0, noloanY);

  // Add explanation of incorrect
  model.addListener(function() {
    layout();
    selRateLabel((truepositive.length+falsepositive.length) / model.items.length);
    truePosLabel(truepositive.length / model.items.length);
    falsePosLabel(falsepositive.length / model.items.length);
    noloanLabel(noloan.length / model.items.length);
    svg.selectAll('.icon').call(defineIcon);
  });
}



function computeCurvesNew(model, tprVal, fprVal, linked_model) {
  [threshold_list, accrate_list, tpr_list] = model.maps;
  
  var values = []; var selection_rates = [];
  var values_dempar = []; var values_eqop = [];

  for (var i = 0; i < threshold_list.length; i += 1){
    selection_rates.push(accrate_list[i])
    var value = (tprVal * tpr_list[i] + fprVal * (accrate_list[i] - tpr_list[i]))
    values.push(model.items.length * value);

    if (!(typeof linked_model == "undefined")) {
      var t1, t2, idx, idx2, threshold_list_linked, accrate_list_linked, tpr_list_linked;
      [threshold_list_linked, accrate_list_linked, tpr_list_linked] = linked_model.maps;

      [t1, idx] = getTranslatedThresh(linked_model, accrate_list, accrate_list[i]);
      
      linked_value = tprVal * tpr_list_linked[idx] + fprVal * (tpr_list_linked[idx] - tpr_list_linked[idx])
      values_dempar.push(value + linked_model.items.length * linked_value)

      [t2, idx2] = getTranslatedThresh(linked_model, tpr_list, tpr_list[i])
      linked_value2 = tprVal * tpr_list_linked[idx2] + fprVal * (tpr_list_linked[idx2] - tpr_list_linked[idx2])
      values_eqop.push(value + linked_model.items.length * linked_value2)
    }
  }
  return [selection_rates, values, values_dempar, values_eqop]
}

function computeCurves(model, tprVal, fprVal, linked_model, stepSize = 1) {
  //if (!(typeof linked_model == "undefined"))
  var values = []; var selection_rates = [];
  var values_dempar = []; var values_eqop = [];
  for (var t = 0; t <= 100; t += stepSize){
    model.classify(t);
    selection_rates.push(model.positiveRate)
    var value = profit(model.items, tprVal, fprVal)
    values.push(value);
 
    if (!(typeof linked_model == "undefined")) {
      [t1, idx] = getTranslatedThresh(linked_model, linked_model.maps[1], model.positiveRate);

      linked_model.classify(t1);
      value_dempar = profit(linked_model.items, tprVal, fprVal);
      values_dempar.push(value + value_dempar);
    }
  }

  // this is horrible, how do we do sequential stuff in javascript
  for (var t = 0; t <= 100; t += stepSize){
    model.classify(t);
    var value = profit(model.items, tprVal, fprVal)

    if (!(typeof linked_model == "undefined")) {
      [t2, idx]  = getTranslatedThresh(linked_model, linked_model.maps[2], model.tpr);
      linked_model.classify(t2)

      value_eqop = profit(linked_model.items, tprVal, fprVal);
      values_eqop.push(value + value_eqop);
    }
  }
  return [selection_rates, values, values_dempar, values_eqop]
}

function displayCurves(id, model, tprVal, fprVal, linked_model) {

  res = computeCurves(model, tprVal, fprVal, linked_model);
  var values = res[1]; var selection_rates = res[0];
  if (!(typeof linked_model == "undefined")) {
    values_dempar = res[2];
    values_eqop = res[3];
  }
  
  var margin = {top: 10, right: 10, bottom: 20, left: 50},
    width = 400 - margin.left - margin.right,
    height = 150 - margin.top - margin.bottom;
      // set the ranges
  var x = d3.scaleLinear().range([0, width]);
  var y = d3.scaleLinear().range([0, height]);

  // define the line
  var valueline = d3.line()
      .x(function(d) { return x(d.x); })
      .y(function(d) { return y(d.y); });

  // append the svg object to the body of the page
  // appends a 'group' element to 'svg'
  // moves the 'group' element to the top left margin
  var svg = d3.select('#' + id).append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");
  
  var maxprof_offset = 0;
  if (!(typeof linked_model == "undefined")) {
    // somehow wasteful to recompute these curves just for max prof
    res = computeCurves(linked_model, tprVal, fprVal);
    maxprof_offset = Math.max.apply(Math, res[1])
  }

  data = []
  for (var i = 0; i < selection_rates.length; i++){
    var d={};
    d.x = selection_rates[i];
    d.y = values[i];
    data.push(d);
  }

    // Scale the range of the data
  x.domain(d3.extent(data, function(d) { return d.x; }));
  y.domain([d3.max(data, function(d) { return d.y + maxprof_offset; }), Math.max(d3.min(data, function(d) { return d.y; }),-60)]);


    // gridlines in x axis function https://bl.ocks.org/d3noob/c506ac45617cf9ed39337f99f8511218
  function make_x_gridlines() {   
      return d3.axisBottom(x)
          .ticks(10)
  }

  // gridlines in y axis function
  function make_y_gridlines() {   
      return d3.axisLeft(y)
          .ticks(5)
  }

  // add the X gridlines
  svg.append("g")     
      .attr("class", "grid")
      .attr("transform", "translate(0," + height + ")")
      .call(make_x_gridlines()
          .tickSize(-height)
          .tickFormat("")
      )

  // add the Y gridlines
  svg.append("g")     
      .attr("class", "grid")
      .call(make_y_gridlines()
          .tickSize(-width)
          .tickFormat("")
      )

    // Add the valueline path.
  var line = svg.append("path")
        .data([data])
        .attr("class", "line")
        .attr("d", valueline);

  var data_dempar, data_eqop, line_dempar, line_eqop, data_maxprof, line_maxprof;
  if (!(typeof linked_model == "undefined")) {
    data_maxprof = []; data_dempar = []; data_eqop = [];
    for (var i = 0; i < selection_rates.length; i++){
      var d={};
      d.x = selection_rates[i];
      d.y = values[i] + maxprof_offset;
      data_maxprof.push(d);
      var d={};
      d.x = selection_rates[i];
      d.y = values_dempar[i];
      data_dempar.push(d);
      var d={};
      d.x = selection_rates[i];
      d.y = values_eqop[i];
      data_eqop.push(d);
    }
    line_maxprof = svg.append("path")
        .data([data_maxprof])
        .attr("class", "line_maxprof")
        .attr("d", valueline);
    line_dempar = svg.append("path")
        .data([data_dempar])
        .attr("class", "line_dempar")
        .attr("d", valueline);
    line_eqop = svg.append("path")
        .data([data_eqop])
        .attr("class", "line_eqop")
        .attr("d", valueline);
  }


  // Sliding threshold point.
  var selection_point = svg.append('circle')
                            .attr('cx', x(selection_rates[50]))
                            .attr('cy', y(values[50])).
      attr('r', 6)

  // var thresholdLabel = svg.append('text').text(values[50])
  //     .attr('x', x(selection_rates[50])+12)
  //     .attr('y', y(values[50]))
  //     .attr('text-anchor', 'left').attr('class', 'bold-label');
      
  // Add the X Axis
  svg.append("g")
        .attr('class', 'histogram-axis')
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x).ticks(10));

  // Add the Y Axis
  svg.append("g")
        .attr('class', 'histogram-axis')
        .call(d3.axisLeft(y).ticks(5));


  function updateThreshold(t) {
    t = Math.max(0, Math.min(t, 99));
    t = Math.round(t);

    //thresholdLabel.attr('x', x(selection_rates[t])+12).attr('y', y(values[t])).text(Math.round(values[t]));
    selection_point.attr('cx', x(selection_rates[t])).attr('cy', y(values[t]));
    svg.selectAll('.icon').call(defineIcon);
  }
  model.addListener(function(event) {
    updateThreshold(model.threshold);
    // if event...
  });
}
 function singleHistogramTable() {
  document.getElementById('single-histogram-table').innerHTML = '<table>\
    <tr>\
      <td>&nbsp;</td>\
      <td colspan=4>\
          <div class="figure-title">\
          Credit score and repayment distribution\
          </div>\
      </td>\
    </tr>\
    <tr>\
      <td valign="top">\
        <table>\
          <tr><td width="200" valign="top" align="right">\
            <div style="margin-top:14px">\
            <span class="margin-bold">Credit Score</span><br>\
            <span class="margin-text">\
            higher scores represent higher likelihood of payback\
            </span>\
            </div>\
          </td></tr>\
          <tr><td width="200" valign="bottom" align="right">\
            <div style="margin-top:140px">\
            <span class="margin-text">\
            each circle represents a person,\
            with dark circles showing people who pay back their\
            loans and light circles showing people who default\
            </span>\
            </div>\
          </td></tr>\
          <tr><td width="200" valign="bottom" align="right">\
            <div style="margin-top:30px">\
            <span class="margin-bold">Color</span>\
            </div>\
          </td></tr>\
          </table>\
      </td>\
      <td valign="top">\
        <div class="big-label" style="margin-bottom:50px"> </div>\
        <div id="single-histogram"></div>\
        <div id="single-histogram-legend" class="histogram-legend"></div>\
      </td>\
    </tr>\
  </table>';
 }
function singleHistogramInteractiveTable() {
  document.getElementById('single-histogram-interactive-table').innerHTML = '<table>\
    <tr>\
      <td>&nbsp;</td>\
      <td colspan=4>\
          <div class="figure-title">\
          Loan thresholds and outcomes\
          </div>\
          <div class="figure-caption">\
          Drag the black threshold bars left or right to change the cut-offs for loans.<br>\
          </div>\
      </td>\
    </tr>\
    <tr>\
      <td valign="top">\
        <table>\
          <tr><td width="200" valign="top" align="right">\
            <div style="margin-top:44px">\
            <span class="margin-bold">Credit Score</span><br>\
            <span class="margin-text">\
            higher scores represent higher likelihood of payback\
            </span>\
            </div>\
          </td></tr>\
          <tr><td width="200" valign="bottom" align="right">\
            <div style="margin-top:160px">\
            <span class="margin-text">\
            each circle represents a person,\
            with dark circles showing people who pay back their\
            loans and light circles showing people who default\
            </span>\
            </div>\
          </td></tr>\
          <tr><td width="200" valign="bottom" align="right">\
            <div style="margin-top:30px">\
            <span class="margin-bold">Color</span>\
            </div>\
          </td></tr>\
\
        </table>\
      </td>\
\
      <td valign="top">\
        <div class="big-label" style="margin-bottom:50px">Threshold Decision</div>\
        <div id="single-histogram0"></div>\
        <div id="single-histogram-legend0" class="histogram-legend"></div>\
      </td>\
      <td width=30>&nbsp;</td>\
      <td valign="top">\
        <div class="big-label" style="margin-bottom:20px;margin-left:10px">Outcome</div>\
        <div id="single-loans0"></div>\
        <div class="profit-readout">Profit: <span class="readout" id="single-profit0"></span><br>\
          Average Credit Score Change: <span class="readout" id="single-scorechange0"></div>\
      </td>\
\
      <td width=5 valign="bottom">&nbsp;\
      </td>\
    </tr>\
  </table>\
';
 }

 function comparisonTable() {
  document.getElementById('comparison-histogram-table').innerHTML = '<div class="figure-title">\
    Simulating loan decisions for different groups\
  </div>\
  <div class="figure-caption">\
    Drag the black threshold bars left or right to change the cut-offs for loans.\
  </div>\
  <table>\
    <tr>\
      <td colspan=3>\
\
      </td>\
    </tr>\
    <tr>\
      <td valign="top">\
        <div class="big-label">Blue Population</div>\
        <br><br>\
        <div id="histogram0_nocurves"></div>\
        <div id="histogram-legend0_nocurves" class="histogram-legend"></div>\
        <div class="profit-readout"> Average Credit Score Change: <span class="readout" id="comparison0_chg"></div>\
      </td>\
      <td width=20>&nbsp;</td>\
      <td valign="top">\
        <!-- start upper right content label -->\
        <div class="big-label" style="margin-left:10px">Red Population</div>\
        <!-- end upper right content label -->\
        <br><br>\
        <!-- start upper right content -->\
        <div id="histogram1_nocurves"></div>\
        <div id="histogram-legend1_nocurves" class="histogram-legend"></div>\
        <!-- end upper right content -->\
        <div class="profit-readout"> Average Credit Score Change: <span class="readout" id="comparison1_chg"></div>\
      </td>\
    </tr>\
    <tr>\
      <td valign="top">\
\
         <div id="profit-title" style="position:relative">\
           Total profit = <span id="total-profit_nocurves"></span>\
            <div style="position:absolute;left:-20;top:-20">\
            <svg>\
            <rect class="annotation max-profit-annotation"\
             x="10" y="10" width="230" height="50" rx="20" ry="20"\
            />\
            </svg>\
            </div>\
          </div>\
      </td>\
      <td width=20>&nbsp;</td>\
      <td valign="top">\
      </td>\
    </tr>\
  </table>'; }
function comparisonCurvesTable() {
  document.getElementById('comparison-curves-table').innerHTML = '  <div class="figure-title">\
    Simulating loan decisions for different groups\
  </div>\
  <div class="figure-caption">\
    Drag the black threshold bars left or right to change the cut-offs for loans.<br>\
    Click on different preset loan strategies.\
  </div>\
  <table>\
    <tr>\
      <td rowspan=4 width=200 valign="top">\
        <div class="big-label" style="margin-top:3px">Loan Strategy</div>\
        <span class="margin-text">\
          Maximize profit with:\
          <br><br><br>\
          <button class="demo" id="max-profit">MAX PROFIT</button>\
          <br>No constraints\
          <p><br>\
          <button class="demo" id="demographic-parity">DEMOGRAPHIC PARITY</button>\
          <br>\
          Same fractions blue / red loans\
          <p><br>\
          <button class="demo" id="equal-opportunity">EQUAL OPPORTUNITY</button>\
          <br>\
          Same fractions blue / red loans<br>\
          to people who can pay them off\
          <br><br><br>\
\
          \
        </span>\
      </td>\
      <td colspan=3>\
\
      </td>\
    </tr>\
    <tr>\
      <td valign="top">\
        <div class="big-label">Blue Population</div>\
        <br><br>\
        <div id="histogram0"></div>\
        <div id="histogram-legend0" class="histogram-legend"></div>\
      </td>\
      <td width=20>&nbsp;</td>\
      <td valign="top">\
        <!-- start upper right content label -->\
        <div class="big-label" style="margin-left:10px">Red Population</div>\
        <!-- end upper right content label -->\
        <br><br>\
        <!-- start upper right content -->\
        <div id="histogram1"></div>\
        <div id="histogram-legend1" class="histogram-legend"></div>\
        <!-- end upper right content -->\
      </td>\
    </tr>\
    <tr>\
      <td valign="top">\
         <div id="profit-title" style="position:relative">\
           Total profit = <span id="total-profit"></span>\
            <div style="position:absolute;left:-20;top:-20">\
            <svg>\
            <rect class="annotation max-profit-annotation"\
             x="10" y="10" width="230" height="50" rx="20" ry="20"\
            />\
            </svg>\
            </div>\
          </div>\
      </td>\
      <td width=20>&nbsp;</td>\
      <td valign="top">\
      </td>\
    </tr>\
    <tr>\
      <td valign="top" width=400>\
        <div class="big-label" style="margin-bottom:20px;margin-left:10px">Average Credit Score Change</div>\
        <div id="outcomes0">\
        \
        </div>\
        <div class="big-label" style="margin-bottom:20px;margin-left:10px">Bank Profit</div>\
        <div id="profits0">\
                      \
        </div>\
      </td>\
      <td width=20>&nbsp;</td>\
      <td valign="top" width=400>\
        <div class="big-label" style="margin-bottom:20px;margin-left:10px">Average Credit Score Change</div>\
        <div id="outcomes1">\
        \
        </div>\
        <div class="big-label" style="margin-bottom:20px;margin-left:10px">Bank Profit</div>\
        <div id="profits1">\
                      \
        </div>\
      </td>\
\
    </tr>\
  </table>'; }

function singleCurvesTable() {
  document.getElementById('single-curves-table').innerHTML = '  <table>\
    <tr>\
      <td>&nbsp;</td>\
      <td colspan=4>\
          <div class="figure-title">\
          Loan thresholds and outcomes\
          </div>\
          <div class="figure-caption">\
          Drag the black threshold bars left or right to change the cut-offs for loans.<br>\
          </div>\
      </td>\
    </tr>\
    <tr>\
      <td valign="top">\
        <table>\
          <tr><td width="200" valign="top" align="right">\
            <div style="margin-top:44px">\
            <span class="margin-bold">Credit Score</span><br>\
            <span class="margin-text">\
            higher scores represent higher likelihood of payback\
            </span>\
            </div>\
          </td></tr>\
          <tr><td width="200" valign="bottom" align="right">\
            <div style="margin-top:160px">\
            <span class="margin-text">\
            each circle represents a person,\
            with dark circles showing people who pay back their\
            loans and light circles showing people who default\
            </span>\
            </div>\
          </td></tr>\
          <tr><td width="200" valign="bottom" align="right">\
            <div style="margin-top:30px">\
            <span class="margin-bold">Color</span>\
            </div>\
          </td></tr>\
\
        </table>\
      </td>\
\
      <td valign="top">\
        <div class="big-label" style="margin-bottom:50px">Threshold Decision</div>\
        <div id="single-histogram1"></div>\
        <div id="single-histogram-legend1" class="histogram-legend"></div>\
      </td>\
      <td width=30>&nbsp;</td>\
      <td valign="top">\
        <div class="big-label" style="margin-bottom:20px;margin-left:10px">Outcome</div>\
        <div id="single-loans1"></div>\
        <div class="profit-readout">Profit: <span class="readout" id="single-profit1"></span><br>\
          Average Credit Score Change: <span class="readout" id="single-scorechange1"></div>\
      </td>\
    </tr>\
    <tr>\
      <td valign="bottom">&nbsp;\
      </td>\
      <td valign="top" width=400>\
        <div class="big-label" style="margin-bottom:20px;margin-left:10px">Average Credit Score Change</div>\
        <div id="single-outcomes">\
          \
\
        </div>\
        <div class="big-label" style="margin-bottom:20px;margin-left:10px">Bank Profit</div>\
        <div id="single-profits">\
                        \
        \
        </div>\
      </td>\
      <td></td>\
      <td width=100 valign="bottom">\
        <span class="margin-bold">Selection Rate</span><br>\
            <span class="margin-text">\
            loan thresholds are equivalent to selection rates\
            </span>\
      </td>\
    </tr>\
  </table>'; }

  window.onload = main