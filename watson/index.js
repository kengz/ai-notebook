var watson = require('watson-developer-cloud');

var alchemy_language = watson.alchemy_language({
  api_key: 'bca0d43ee6b60c47e9e5eae1cd23f7487a694bbc'
});

var params = {
  text: 'IBM Watson won the Jeopardy television show hosted by Alex Trebek'
};

// alchemy_language.sentiment(params, function (err, response) {
//   if (err)
//     console.log('error:', err);
//   else
//     console.log(JSON.stringify(response, null, 2));
// });

console.log(watson)

// sample Alchemy API curl:
// curl 'http://gateway-a.watsonplatform.net/calls/text/TextGetEmotion?apikey=7469f5efe506f635859a2a0025d5797ff8d5e41f&text=Thisisreallyquiteshitayy&outputMode=json'