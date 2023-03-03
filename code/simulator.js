#!/usr/bin/env node

const {readFileSync, writeFileSync, existsSync} = require('fs')
const mqtt = require('mqtt')
require('dotenv').load()
const R = require('ramda')
//require('ofe').call()
const v8 = require('v8')

const {getTaskProfile, gaussianRandom, scaleFeatures, reasonState, reasonReward, updateStateFeaturesToNextState} = require('./utils')
//const {createCML, createConsolidatedCML, getInstructions, createRL_CML} = require('./CMLGenerator')
/*const {
    createJob,
    createJobWithCSVToJSON,
    createUnit,
    configureJob,
    launchToTheDemandWorkforce,
    launchInternalChannel,
    launchODWAndInternalChannel,
    createTestQuestions,
    uploadTestQuestions,
    convertUploadedTestQuestions
} = require('./CrowdSourcingClient')

const km = require('./K-Means') */

function ratePerformance(task, allpoints, eDistance, mode) {
    const guidelineCentroids = {}
    const currentInput = []
    const stateKeys = Object.keys(task.states)
    stateKeys.forEach((key) => {
        if(task.states[key][key].IsGoal == true) {
        const expr = R.replace(/\)/g, '', R.replace(/\(/g, '', task.states[key][key].HasExpression))
        const tokens = expr.split(" ")
            const toks = tokens.map((t) => {
                const to = t.replace(/\+|\-|\/|\*/g, "_")
                return to
            })
        if (tokens.length > 3) {
            const res = parseComplexExpression(task, toks).result
            //const feats = Object.keys(res.unscaledVector)
            guidelineCentroids[key] = res.unscaledVector

        } else {
            const res = parseExpression(task, toks).result
            //const feats = Object.keys(res)
            guidelineCentroids[key] = res.unscaledVector
        }
    }
    })

    const rating = computeRatingsForClusters(5, allpoints, task, 100, guidelineCentroids, 5, eDistance, mode)
    return rating
}

function computeRatingsForClusters(numClusters, allpoints, task, numIterations, guidelineCentroids, numRatings, eDistance, mode) {
    let gl = []
        for (let g in guidelineCentroids) {
            const ar = []
            for(let feat in task.features) {
                if (feat in guidelineCentroids[g])
                    ar.push(guidelineCentroids[g][feat])
                else
                    ar.push(0)
            }
            gl.push(ar)
    }
    const activity = {}
    const points = []
    for(let act in allpoints) {
        allpoints[act].points.forEach((point) => {
            if(point.length > 0)
                points.push(point)
        })
    }
    let distances = []
    if(mode) {
        const cluster = km.kmodes(numClusters, points, numIterations)
        const result = km.selectBestCluster(cluster)
        console.log("K-Means Cost: ", result.cost)
        gl.forEach((gPoint) => {
            result.centroids.forEach(function (c) {
                const distance = km.hammingDistance(c.coordinates, gPoint)
                if(!(distances.includes(distance)))
                    distances.push(distance)
            })
        })

    } else {
        const cluster = km.kmeans(numClusters, points, numIterations)
        const result = km.selectBestCluster(cluster)
        console.log("K-Means Cost: ", result.cost)
        gl.forEach((gPoint) => {
            result.centroids.forEach(function (c) {
                const distance = km.euklidianDistance(c.coordinates._data, gPoint)
                if(!(distances.includes(distance)))
                    distances.push(distance)

            })
        })
    }
    distances.sort((a,b) => {
        return a - b
    })
    eDistance.sort((a,b) => {
        return a - b
    })
    var flag = false
    var rate = 0
    if(eDistance.length >= numRatings) {
        let factor = 0
        distances.forEach((dist) => {
            var pos = 0
            var mod = eDistance.length % numRatings
            var range = (eDistance.length - mod) / numRatings
            for(let e = 0; e < eDistance.length; e++) {
                if (dist <= eDistance[e] /*&& !flag*/) {
                    flag = true
                    pos = e + 1
                    if(factor >= 1)
                        rate += (numRatings - pos + 1) / range
                    else
                        rate = (numRatings - pos + 1) / range
                    factor += 1
                }
            }
        })
        rate /= factor
        if(flag)
            activity['rating'] = rate
        else
            activity['rating'] = 0
    } else {
        activity['rating'] = 0
    }
    distances.forEach((dist) => {
        if(!(eDistance.includes(dist)))
            eDistance.push(dist)
    })

    return activity
}

function selectBiggestCluster(clusters) {
    let size = 0
    let cluster = []
    clusters.centroids.forEach((centroid) => {
        console.log(centroid.datapoints.length)
        if(size <= centroid.datapoints.length) {
            size = centroid.datapoints.length
            if(cluster.length > 0) {
                let index = cluster.length
                while(index--) {
                    if(cluster[index].datapoints.length < size) {
                        cluster.splice(index, 1)
                    }
                }
            }
            cluster.push(centroid)
        }
    })
    return cluster
}

function loadModel(filepath) {
    if(existsSync(filepath)) {
        let model = readFileSync(filepath)
        model = JSON.parse(model.toString())
        return model
    } else return undefined
}

async function connectToIoTBroker(MQTT_HOST, subscriptionTopic, task) {
    const client =  await mqtt.connect(MQTT_HOST);
    let flag = false
    let lastState = {scaledVector: {}, unscaledVector: {}}
    let lastSendState = {scaledVector: {}, unscaledVector: {}}
    const featStates = {}
    for(let topic in task.topics) {
        if(!topic.includes("Request")) {
            if (!(topic in featStates))
                featStates[topic] = []
            for (let s in task.states) {
                for (let feat in task.topics[topic][topic].HasObservationFeature) {
                    if (feat in task.states[s][s].HasObservationFeature /*&& task.states[s][s].IsGoal == false*/) {
                        if (!featStates[topic].includes(s))
                            featStates[topic].push(s)
                    }
                }
            }
        }
    }
    return new Promise((resolve, reject) => {client.on("connect", () => {
        client.subscribe(subscriptionTopic)
        //Generate random state and send to agent
        if(!flag) {
            flag = true
            const generateNewState = generateInitialState(task) //generateRandomState(task, featStates)
            const initialState = generateNewState(false)
            const featKeys = Object.keys(task.features)
            featKeys.forEach((feat) => {
                if(task.features[feat][feat].HasFeatureType.includes("AGG")) {
                    for(let f in task.features[feat][feat].HasObservationFeature) {
                        if(f in initialState.unscaledVector) {
                            lastSendState.unscaledVector[f] = initialState.unscaledVector[f]
                            lastSendState.scaledVector[f] = initialState.scaledVector[f]
                            lastState.unscaledVector[f] = initialState.unscaledVector[f]
                            lastState.scaledVector[f] = initialState.scaledVector[f]
                        } else {
                            if(f in task.features[feat][feat].HasObservationFeature) {
                                const min = task.features[feat][feat].HasObservationFeature[f].HasHasRangeStart
                                const max = task.features[feat][feat].HasObservationFeature[f].HasRangeEnd
                                const range = max - min
                                let val = undefined
                                if(task.features[feat][feat].HasFeatureType === "NOMINAL") {
                                    val = getRandomInt(0, 2)
                                } else {
                                    val = getRandomDouble(min, max)
                             }
                                lastState.unscaledVector[f] = val
                                lastState.scaledVector[f] = scaleFeatures(val, min, max)
                                lastSendState.unscaledVector[f] = val
                                lastSendState.scaledVector[f] = scaleFeatures(val, min, max)
                            }
                        }
                    }
                } else {
                    if (feat in initialState.unscaledVector) {
                        lastSendState.unscaledVector[feat] = initialState.unscaledVector[feat]
                        lastSendState.scaledVector[feat] = initialState.scaledVector[feat]
                        lastState.unscaledVector[feat] = initialState.unscaledVector[feat]
                        lastState.scaledVector[feat] = initialState.scaledVector[feat]
                    } else {
                        const min = task.features[feat][feat].HasRangeStart
                        const max = task.features[feat][feat].HasRangeEnd
                        const range = max - min
                        let val = undefined
                        if(task.features[feat][feat].HasFeatureType === "NOMINAL") {
                            val = 0 //getRandomInt(0, 2)
                        } else {
                            val = getRandomDouble(min, max)
                        }
                        lastState.unscaledVector[feat] = val
                        lastState.scaledVector[feat] = scaleFeatures(val, min, max)
                        lastSendState.unscaledVector[feat] = val
                        lastSendState.scaledVector[feat] = scaleFeatures(val, min, max)
                    }
                }
            })
            const topicKeys = Object.keys(task.topics)
            const justTopics = topicKeys.filter((t) => {return t !== 'RequestTopic'})
            justTopics.forEach(key => {
                const state = {scaledVector: {}, unscaledVector: {}}
                const topic = task.topics[key][key]
                const topicFeatures = Object.keys(topic.HasObservationFeature)
                const feats = topicFeatures.filter(feat => (feat in initialState.unscaledVector))
                feats.forEach((f) => {
                    if (topicFeatures.includes(f)) {
                        feats.forEach(feat => {
                            state.scaledVector[feat] = initialState.scaledVector[feat]
                            state.unscaledVector[feat] = initialState.unscaledVector[feat]
                        })
                        const topicName = topic.HasName
                    }
                })



            const freezedState = state
            const sendMessage = JSON.stringify({
                timestamp: new Date(),
                state: freezedState,
                status: "unknown",
                reward: 0,
                lastState: lastState,
                lastSendState: lastSendState,
                lastAction: undefined
            })
            client.publish(topic.HasName /*task.id*/, sendMessage, {qos: 2, retain: false})
        })
        }
            resolve({client: client, lastState: lastState, lastSendState : lastSendState})
        })
    })
}

function generateRandomState(task, featStates) {
    //let index = 0
    const states = []//Object.keys(task.states)
    return (goals) => {
        const updateStates = (goals !== undefined && goals !== null) ? states.filter(state => goals.indexOf(state) === -1): states
        /*if(index >= updateStates.length) {
            index = 0
        }*/
        for(let f in featStates) {
            let filtered = (goals !== undefined && goals !== null) ?
                featStates[f].filter(filt => !(goals.includes(filt))) : featStates[f]
            let index = getRandomInt(0, filtered.length)
            if(filtered.length > 0 && filtered[index] != undefined)
                states.push(filtered[index])
        }
        const res = {scaledVector: {}, unscaledVector: {}}
        for(let st of states) {
            const state = task.states[st][st]
            const expression = R.replace(/\)/g, '', R.replace(/\(/g, '', state.HasExpression))
            const tokens = expression.split(" ")
            const toks = tokens.map((t) => {
                const to = t.replace(/\+|\-|\/|\*/g, "_")
                return to
            })
            const result = toks.indexOf("AND") !== -1 || toks.indexOf("OR") !== -1 || toks.indexOf("XOR") !== -1 ? parseComplexExpression(task, toks).result : parseExpression(task, toks).result
            for(let val in result.unscaledVector) {
                res.unscaledVector[val] = result.unscaledVector[val]
                res.scaledVector[val] = result.scaledVector[val]
            }
            //const nullFeats = Object.keys(task.features)
            // nullFeats.forEach((nf) => {

            /*  if (!(nf in lastState.unscaledVector) && !(nf in result.unscaledVector)) {
                  const min = task.features[nf][nf].HasRangeStart
                  const max = task.features[nf][nf].HasRangeEnd
                  lastState.unscaledVector[nf] = getRandomDouble(min, max)
                  lastState.scaledVector[nf] = scaleFeatures(lastState.unscaledVector[nf], min, max)
              }

              if(nf in result.unscaledVector) {
                  lastState.unscaledVector[nf] = result.unscaledVector[nf]
                  lastState.scaledVector[nf] = result.scaledVector[nf]
              } */
            // })
            //lastSendState = result
            //index += 1
        }
        return res
    }
}

function generateInitialState(task) {
    let index = 0
    const initalStates = []
    for (let state in task.states) {
        if (task.states[state][state].IsInitialState) {
            initalStates.push(task.states[state][state])
        }
    }
    return (goal) => {
        const res = {scaledVector: {}, unscaledVector: {}}
        const featKeys = Object.keys(task.features)
        for(let f of featKeys) {
            res.scaledVector[f] = 0
            res.unscaledVector[f] = 0
        }
        //console.log(initalStates.length)
        if(goal && index < initalStates.length - 1)
            index += 1
        else if(goal && index >= initalStates.length - 1)
            index = 0
        if(index <  0)
            index = 0
        const init = initalStates[index]
        const expression = init.HasExpression
        let brackless = undefined
        if(expression.includes("XOR")) {
            const xors = expression.split('XOR')
            let pos = getRandomInt(0, xors.length)
            if(pos < 0)
                pos = 0
            brackless = R.replace(/\)|\(/g, '', xors[pos]).trim()
        } else {
            brackless = R.replace(/\)/g, '', R.replace(/\(/g, '', expression))
        }
        const tokens = brackless.split(" ")
        const toks = tokens.map((t) => {
            const to = t.replace(/\+|\-|\/|\*/g, "_")
            return to
        })
        const result = toks.indexOf("AND") !== -1 || toks.indexOf("OR") !== -1 || toks.indexOf("XOR") !== -1 ? parseComplexExpression(task, toks).result : parseExpression(task, toks).result
        for(let val in result.unscaledVector) {
            res.unscaledVector[val] = result.unscaledVector[val]
            res.scaledVector[val] = result.scaledVector[val]
        }
        return res
    }
}

/* function updateStateFeaturesToNextState(task, statevector, action) {
    let featVector = {}
    let scaledFeatVector = {}
    let featRanges = {}
    let statename = reasonState(task, statevector.unscaledVector)
    statename.forEach((sta) => {
        let state = task.states[sta][sta]
        let ruletokens = state.HasExpression.split(" ")
        let index = 0
        ruletokens.forEach((token) => {
            if (index % 4 === 0) {
                featRanges[token] = {min: undefined, max: undefined}
                if (ruletokens[index + 1] === ">" || ruletokens[index + 1] === ">=") {
                    featRanges[token].min = parseFloat(ruletokens[index + 2])
                } else if (ruletokens[index + 1] === "<" || ruletokens[index + 1] === "<=") {
                    featRanges[token].max = parseFloat(ruletokens[index + 2])
                } else if (ruletokens[index + 1] === "==") {
                    featRanges[token].min = parseFloat(ruletokens[index + 2])
                    featRanges[token].max = parseFloat(ruletokens[index + 2])
                }
            }
            index += 1
        })
        const feats = Object.keys(featRanges)
        feats.forEach((f) => {
            if (featRanges[f].min === undefined) {
                const mini = parseFloat(task.features[f][f].HasRangeStart)
                featRanges[f].min = mini
            }
            if (featRanges[f].max === undefined) {
                const maxi = parseFloat(task.features[f][f].HasRangeEnd)
                featRanges[f].max = maxi
            }
        })
    })
    const effects = action.HasEffect
    const effectKeys = Object.keys(effects)
    const stateKeys = Object.keys(statevector.unscaledVector)
    effectKeys.forEach((ek) => {
        if (effects[ek].HasObservationFeature !== null && effects[ek].HasObservationFeature !== undefined) {
            const featKeys = Object.keys(effects[ek].HasObservationFeature)
            const impactType = effects[ek].HasImpactType
            const impactVal = effects[ek].HasImpactRange
            stateKeys.forEach((sk) => {
                if (featKeys.indexOf(sk) !== -1) {
                    const min = task.features[sk][sk].HasRangeStart
                    const max = task.features[sk][sk].HasRangeEnd
                    if (statevector.unscaledVector[sk] < min) {
                        //const range = max - min
                        //const val = max - (range / 2)
                        statevector.unscaledVector[sk] = min
                        statevector.scaledVector[sk] = scaleFeatures(min, min, max)
                    }
                    if (statevector.unscaledVector[sk] > max) {
                        statevector.unscaledVector[sk] = max
                        statevector.scaledVector[sk] = scaleFeatures(max, min, max)
                    }
                    if (sk in featRanges) {
                        const featMin = featRanges[sk].min
                        const featMax = featRanges[sk].max
                        if (impactType === "INCREASE") {
                            const val = featMax + impactVal
                            featVector[sk] = val > max ? max : val
                        } else if (impactType === "DECREASE") {
                            const val = featMin - impactVal
                            featVector[sk] = val < min ? min : val
                        } else if (impactType === "CONVERT") {
                            featVector[sk] = statevector.unscaledVector[sk] === 1 ? 0 : 1
                        }
                        scaledFeatVector[sk] = scaleFeatures(featVector[sk], min, max)
                    }
                }
                else if (!(sk in featVector)) {
                    featVector[sk] = statevector.unscaledVector[sk]
                    scaledFeatVector[sk] = statevector.scaledVector[sk]
                }
            })
        }
        else {
            featVector = statevector.unscaledVector
            scaledFeatVector = statevector.scaledVector
        }

    })
    return {scaledVector : scaledFeatVector, unscaledVector : featVector}
} */

function parseComplexExpression(task, tokens) {
    const copiedTokens = tokens.slice()
    let index = 0
    const result = {scaledVector: {}, unscaledVector: {}}
    const ranges = {}
    const operators = {}
    copiedTokens.forEach((token) => {
        if(index % 4 === 0) {
            if (index + 2 < copiedTokens.length) {
                if(token in task.features) {
                if (!(token in ranges))
                    ranges[token] = {min: undefined, max: undefined, composition: {and: [], or: [], xor: []}}
                if (index + 4 < copiedTokens.length) {
                    if (copiedTokens[index + 3] === "AND" && ranges[token].composition.and.indexOf(copiedTokens[index + 4]) === -1)
                        ranges[token].composition.and.push(copiedTokens[index + 4])
                    else if (copiedTokens[index + 3] === "OR" && ranges[token].composition.or.indexOf(copiedTokens[index + 4]) === -1)
                        ranges[token].composition.or.push(copiedTokens[index + 4])
                    else if (copiedTokens[index + 3] === "XOR" && ranges[token].composition.xor.indexOf(copiedTokens[index + 4]) === -1)
                        ranges[token].composition.xor.push(copiedTokens[index + 4])
                }
                if (copiedTokens[index + 1].indexOf(">") !== -1 && copiedTokens[index + 1].indexOf("=") == -1) {
                    ranges[token]["min"] = parseFloat(copiedTokens[index + 2]) + 1
                } else if (copiedTokens[index + 1].indexOf(">=") !== -1) {
                    ranges[token]["min"] = parseFloat(copiedTokens[index + 2])
                } else if (copiedTokens[index + 1].indexOf("<") !== -1 && copiedTokens[index + 1].indexOf("=") == -1) {
                    ranges[token]["max"] = parseFloat(copiedTokens[index + 2]) - 1
                } else if (copiedTokens[index + 1].indexOf("<=") !== -1) {
                    ranges[token]["max"] = parseFloat(copiedTokens[index + 2])
                } else if (copiedTokens[index + 1].indexOf("==") !== -1) {
                    ranges[token]["max"] = parseFloat(copiedTokens[index + 2])
                    ranges[token]["min"] = parseFloat(copiedTokens[index + 2])
                }
            }
        }
        }
        index += 1
    })
    const keys = Object.keys(ranges)
    keys.forEach((k) => {
        if(ranges[k].min === undefined) {
            if(k in task.features) {
                ranges[k].min = task.features[k][k].HasRangeStart
            } else {
                for(let f in task.features) {
                    if(task.features[f][f].HasFeatureType.includes("AGG")) {
                        if(k in task.features[f][f].HasObservationFeature) {
                            ranges[k].min = task.features[f][f].HasObservationFeature[k].HasRangeStart
                        }
                    }
                }
            }
        }
        if(ranges[k].max === undefined) {
            if(k in task.features) {
                ranges[k].max = task.features[k][k].HasRangeEnd
            } else {
                for(let f in task.features) {
                    if(task.features[f][f].HasFeatureType.includes("AGG")) {
                        if(k in task.features[f][f].HasObservationFeature) {
                            ranges[k].max = task.features[f][f].HasObservationFeature[k].HasRangeEnd
                        }
                    }
                }
            }
        }
    })

    keys.forEach((k) => {
        if(!(k in result.unscaledVector)) {
            const operand = ranges[k]
            const min = operand.min
            const max = operand.max
            let val = undefined
            if(task.features[k][k].HasFeatureType === "NOMINAL") {
                val = getRandomInt(min, max)
            } else {
                val = getRandomDouble(min, max)
            }
            result.unscaledVector[k] = val
            result.scaledVector[k] = scaleFeatures(val, min, max)
            if (operand.composition.and.length > 0) {
                operand.composition.and.forEach(cp => {
                    if ((cp in task.features)) {
                        const instance = ranges[cp]
                        let val2 = undefined
                        if (task.features[k][k].HasFeatureType === "NOMINAL") {
                            val2 = getRandomInt(instance.min, instance.max)
                        } else {
                            val2 = getRandomDouble(instance.min, instance.max)
                        }
                        result.unscaledVector[cp] = val2
                        result.scaledVector[cp] = scaleFeatures(val2, instance.min, instance.max)
                    }
                })
            }
            if (operand.composition.or.length > 0) {
                operand.composition.or.forEach(cp => {
                    if ((cp in task.features)) {
                    const instance = ranges[cp]
                    let val2 = undefined
                    if (task.features[k][k].HasFeatureType === "NOMINAL") {
                        val2 = getRandomInt(instance.min, instance.max)
                    } else {
                        val2 = getRandomDouble(instance.min, instance.max)
                    }

                    result.unscaledVector[cp] = val2
                    result.scaledVector[cp] = scaleFeatures(val2, instance.min, instance.max)
                }
                })

            }
            if (operand.composition.xor.length > 0) {
                operand.composition.xor.forEach(cp => {
                    if ((cp in task.features)) {
                        const instance = ranges[cp]
                        const range1 = (instance.max * 2) - instance.max
                        const range2 = instance.min
                        let val2 = undefined
                        if (task.features[k][k].HasFeatureType === "NOMINAL") {
                            val2 = [instance.max, instance.min]
                        } else {
                            val2 = [getRandomDouble(instance.max, instance.max * 2), getRandomDouble(0, instance.min)]
                        }
                        const index = getRandomInt(0, 2)
                        result.unscaledVector[cp] = val2[index]
                        index === 0 ? result.scaledVector[cp] = scaleFeatures(val2[index], instance.max, instance.max * 2) : scaleFeatures(val2[index], 0, instance.min)
                    }
                })
            }
        }
    })
    return {ranges: ranges, result: Object.freeze(result)}
}
function parseExpression(task, tokens) {
    let val = 0
    const copiedTokens = tokens.slice()
    const ranges = {}
    const result = {unscaledVector: {}, scaledVector: {}}
   /* const feats = Object.keys(task.features)
    feats.forEach((feat) => {
        if(!(tokens.includes(feat))) {
            result.unscaledVector[feat] = 0
            result.scaledVector[feat] = 0
        }
    })*/
    if(copiedTokens[0] in task.features) {
        const min = parseFloat(task.features[copiedTokens[0]][copiedTokens[0]].HasRangeStart)
        const max = parseFloat(task.features[copiedTokens[0]][copiedTokens[0]].HasRangeEnd)
        ranges[copiedTokens[0]] = {min: undefined, max: undefined}
        if (copiedTokens[1] === ">") {
            val = getRandomDouble(parseFloat(copiedTokens[2]) + 1, max)
            ranges[copiedTokens[0]].min = parseFloat(copiedTokens[2]) + 1
            ranges[copiedTokens[0]].max = max

        } else if (copiedTokens[1] === "<") {
            val = getRandomDouble(min, parseFloat(copiedTokens[2]) - 1)
            ranges[copiedTokens[0]].min = parseFloat(min)
            ranges[copiedTokens[0]].max = parseFloat(copiedTokens[2]) - 1

        } else if (copiedTokens[1] === ">=") {
            val = getRandomDouble(parseFloat(copiedTokens[2]), max)
            ranges[copiedTokens[0]].min = parseFloat(copiedTokens[2])
            ranges[copiedTokens[0]].max = parseFloat(max)
        } else if (copiedTokens[1] === "<=") {
            val = getRandomDouble(min, parseFloat(copiedTokens[2]))
            ranges[copiedTokens[0]].min = parseFloat(min)
            ranges[copiedTokens[0]].max = parseFloat(copiedTokens[2])
        } else if (copiedTokens[1] === "==") {
            val = parseFloat(copiedTokens[2])
            ranges[copiedTokens[0]].min = parseFloat(copiedTokens[2])
            ranges[copiedTokens[0]].max = parseFloat(copiedTokens[2])
        }
        const scaled = scaleFeatures(val, min, max)
        result.unscaledVector[copiedTokens[0]] = val
        result.scaledVector[copiedTokens[0]] = scaled
    }
    return {ranges: ranges, result: Object.freeze(result)}
}

/*function reasonState(task, featureVector) {
    const reasonedStates = []
    const featKeys = Object.keys(featureVector)
    const stateKeys = Object.keys(task.states)
    stateKeys.map((val)=>{
        let str = ""
        const expr = task.states[val][val].HasExpression
        const and = R.replace(/AND/g, '&&', expr)
        const or = R.replace(/OR/g, '||', and)
        const ex = R.replace(/XOR/g, '^', or) + ";"
        featKeys.forEach((feat) => {
            if (ex.includes(feat)) {
                str += `${feat} = ${featureVector[feat]}; `
            }
        })
        if(str !== "") {
            str += ex
            const result = eval(str)
            if (result) reasonedStates.push(val)
        }
    })
    return reasonedStates
}*/

async function simulate(task, client, stateRanges, lastSendState, lastState, simulation) {
	let numGoalStates = 0
    for(let s in task.states) {
        if(task.states[s][s].IsGoal == true) {
            numGoalStates += 1
        }
    }
    /*for(let topic in task.topics) {
        if(!topic.includes("Request")) {
        if (!(topic in featStates))
            featStates[topic] = []
        for (let s in task.states) {
            for (let feat in task.topics[topic][topic].HasObservationFeature) {
                if (feat in task.states[s][s].HasObservationFeature) {
                    if (!featStates[topic].includes(s))
                        featStates[topic].push(s)
                }
            }
        }
    }
    }*/
    let lastAction = undefined
    const featureKeys = Object.keys(task.features)
    const generateNewState = generateInitialState(task) //generateRandomState(task, featStates)
    let initialStateSize = 0
    for(const s in task.states) {
        if(task.states[s][s].IsInitialState) {
            initialStateSize += 1
        }
    }
    const actionQueue = []
    let lastGoalSize = 0
    const monitorContext = contextControl(task, 5, stateRanges)

    let reasonSR = undefined
    if(task.sequential)
        reasonSR = getSequentialReward(task, stateRanges, numGoalStates)
    else if(task.communicationType == "Synchronized")
        reasonSR = syncReward(task)
    else if(!task.sequential)
        reasonSR = reasonStateReward(task, stateRanges, numGoalStates)
    let msgCounter = 0
    return new Promise((resolve, reject) => {
    client.on("message", (topic, message) => {
        const msg = JSON.parse(message)
        const step = msg.step
        console.log(step)
        let sendAction = undefined
        if(msg.action !== undefined)
            sendAction = Object.keys(msg.action)[0]
        if (sendAction in task.actions) {
            let goal = false
        let newGoal = false
        let reward = 0
        if (msg.status === 'finish') {
            console.log("FINISH!!!")
            if (!client.disconnecting) {
                client.end()
                resolve(client)
            }
        }
        else if (msg.status === "hyperparamtuning") {
            const result = generateNewState(true)
            const topicKeys = Object.keys(task.topics)
            topicKeys.forEach(key => {
                const state = {scaledVector: {}, unscaledVector: {}}
                const topic = task.topics[key][key]
                const topicFeatures = Object.keys(topic.HasObservationFeature)
                const feats = topicFeatures.filter(feat => (feat in result.unscaledVector))
                feats.forEach((f) => {
                    if (topicFeatures.includes(f)) {
                        feats.forEach(feat => {
                            if (result.scaledVector[feat] < 0) {
                                state.scaledVector[feat] = 0
                                state.unscaledVector[feat] = 0
                            } else {
                                state.scaledVector[feat] = result.scaledVector[feat]
                                state.unscaledVector[feat] = result.unscaledVector[feat]
                            }
                        })
                    }
                })


                let names = undefined
                if (!goal)
                    names = reasonState(task, lastSendState.unscaledVector)
                else {
                    const ups = {unscaledVector: {}, scaledVector: {}}
                    for (let fe in lastSendState.unscaledVector) {
                        if (fe in state.unscaledVector)
                            ups.unscaledVector[fe] = state.unscaledVector[fe]
                        else
                            ups.unscaledVector[fe] = lastSendState.unscaledVector[fe]
                    }
                    names = reasonState(task, /*ups*/lastSendState.unscaledVector)
                }
                if (!task.sequential && task.communicationType !== "Synchronized" && !simulation) {
                    const ctxUpdate = monitorContext(lastSendState.unscaledVector, false)
                    if (ctxUpdate !== undefined && Object.keys(ctxUpdate).length > 0) {
                        for (let el in ctxUpdate) {
                            state.unscaledVector[el] = ctxUpdate[el]
                            state.scaledVector[el] = scaleFeatures(ctxUpdate[el], task.features[el][el].HasRangeStart, task.features[el][el].HasRangeEnd)
                            result.unscaledVector[el] = ctxUpdate[el]
                            result.scaledVector[el] = state.scaledVector[el]
                        }
                    }
                }

                const freezedState = state
                const sendMessage = JSON.stringify(JSON.parse(JSON.stringify({
                    timestamp: new Date(),
                    status: "training",
                    state: freezedState,
                    reward: reward,
                    lastState: lastState,
                    lastSendState: lastSendState,
                    lastAction: undefined,
                    goal: goal,
                    stateNames: names
                })))
                lastAction = undefined
                client.publish(key, sendMessage, {qos: 2, retain: false})
            })

            const upKeys = Object.keys(result.unscaledVector)
            upKeys.forEach((up) => {
                lastState.unscaledVector[up] = result.unscaledVector[up]
                lastState.scaledVector[up] = result.scaledVector[up]
                lastSendState.unscaledVector[up] = result.unscaledVector[up]
                lastSendState.scaledVector[up] = result.scaledVector[up]
            })

        }
        else if (msg.status === "training" || msg.status === "eval") {
            const action = msg.action
            msgCounter += 1
            const a = Object.keys(msg.action)[0]
            if (a.length > 0)
                actionQueue.push(a)
            if ((task.communicationType == "Synchronized" && msgCounter == task.numActors) || task.communicationType == "Asynchronized") {
                const lastStates = reasonState(task, lastState.unscaledVector)
              //  console.log(lastState.unscaledVector)
                //const actionId = Object.keys(msg.action)[0]
                //const goalStates = lastStates.filter(elem => task.states[elem][elem].IsGoal == true)
                let result = undefined
                const tempState = JSON.parse(JSON.stringify(lastState))//{scaledVector: {}, unscaledVector: {}}
                /*for (const t in lastState.unscaledVector) {
                    tempState.unscaledVector[t] = lastState.unscaledVector[t]
                    tempState.scaledVector[t] = lastState.scaledVector[t]
                }*/
                const accu = {unscaledVector: {}, scaledVector: {}}
                for (const actionId of actionQueue) {
                    const updated = updateStateFeaturesToNextState(task, tempState, task.actions[actionId][actionId], stateRanges)
                    for (const f in updated.unscaledVector) {
                        if(f in task.features) {
                            accu.unscaledVector[f] = updated.unscaledVector[f]
                            accu.scaledVector[f] = updated.scaledVector[f]
                        }
                    }
                }

                result = JSON.parse(JSON.stringify(accu))
                const tempVec = {unscaledVector: {}, scaledVector: {}}
                for (const f in lastState.unscaledVector) {
                    if (f in result.unscaledVector) {
                        tempVec.unscaledVector[f] = result.unscaledVector[f]
                        tempVec.scaledVector[f] = result.scaledVector[f]
                    } else {
                        tempVec.unscaledVector[f] = lastState.unscaledVector[f]
                        tempVec.scaledVector[f] = lastState.scaledVector[f]
                    }
                }
                if(task.sequential) {
                    const s = reasonState(task, tempVec.unscaledVector)
                    if(lastStates.length > s.length || s.length < 1) {
                        for(const c in lastSendState.unscaledVector) {
                            tempVec.unscaledVector[c] = lastSendState.unscaledVector[c]
                            tempVec.scaledVector[c] = lastSendState.scaledVector[c]
                            result.unscaledVector[c] = lastSendState.unscaledVector[c]
                            result.scaledVector[c] = lastSendState.scaledVector[c]
                        }
                    }
                }
                if (task.communicationType == "Synchronized")
                    reward = reasonSR(tempVec)
                else
                    reward = reasonSR(lastState, action)

                /* for (let f in lastSendState.unscaledVector) {
                     if (f in result.unscaledVector)
                         tempVec.unscaledVector[f] = result.unscaledVector[f]
                     else
                         tempVec.unscaledVector[f] = lastSendState.unscaledVector[f]
                 }*/

                const resStates = reasonState(task, tempVec.unscaledVector)

                const cumulativeGoals = resStates.filter((s) => task.states[s][s].IsGoal)

                const finalState = resStates.filter((s) => task.states[s][s].IsFinalState)

                if (numGoalStates <= 0 || cumulativeGoals.length >= numGoalStates || (finalState !== undefined && finalState.length > 0)) {
                    goal = true
                }
                if (step <= 0 || (task.communicationType == "Asynchronized" && goal && numGoalStates > 0) || (task.sequential && lastGoalSize < cumulativeGoals.length)) {
                    const updateContextState = generateNewState(true)
                    for(let r in result.unscaledVector) {
                        if(result.unscaledVector[r] < 0) {
                            lastState.unscaledVector[r] = 0
                            lastState.scaledVector[r] = 0
                            lastSendState.unscaledVector[r] = 0
                            lastSendState.scaledVector[r] = 0
                        } /*else {
                            lastState.unscaledVector[r] = result.unscaledVector[r]
                            lastState.scaledVector[r] = result.scaledVector[r]
                            lastSendState.unscaledVector[r] = result.unscaledVector[r]
                            lastSendState.scaledVector[r] = result.scaledVector[r]
                        }*/
                    }


                    for (const key in lastState.unscaledVector) {
                        if (lastState.unscaledVector[key] < 0) {
                            lastState.unscaledVector[key] = 0
                            lastSendState.unscaledVector[key] = 0
                            lastState.scaledVector[key] = 0
                            lastSendState.scaledVector[key] = 0
                        }
                    }
                    result = JSON.parse(JSON.stringify(updateContextState))
                    if (goal)
                        lastGoalSize = 0
                    else
                        lastGoalSize = cumulativeGoals.length
                    if (!task.sequential && !simulation)
                        monitorContext(lastSendState.unscaledVector, true)
                }


                const topicKeys = Object.keys(task.topics)
                let idx = 0

                topicKeys.forEach(key => {
                    const state = {scaledVector: {}, unscaledVector: {}}
                    const topic = task.topics[key][key]
                    const topicFeatures = Object.keys(topic.HasObservationFeature)
                    const feats = topicFeatures.filter(feat => (feat in result.unscaledVector))
                    feats.forEach((f) => {
                        if (topicFeatures.includes(f)) {
                            feats.forEach(feat => {
                                if (result.scaledVector[feat] < 0) {
                                    state.scaledVector[feat] = 0
                                    state.unscaledVector[feat] = 0
                                } else {
                                    state.scaledVector[feat] = result.scaledVector[feat]
                                    state.unscaledVector[feat] = result.unscaledVector[feat]
                                }
                            })
                        }
                    })


                    let names = undefined
                    if (!goal)
                        names = reasonState(task, lastSendState.unscaledVector)
                    else {
                        const ups = {unscaledVector: {}, scaledVector: {}}
                        for (let fe in lastSendState.unscaledVector) {
                            if (fe in state.unscaledVector)
                                ups.unscaledVector[fe] = state.unscaledVector[fe]
                            else
                                ups.unscaledVector[fe] = lastSendState.unscaledVector[fe]
                        }
                        names = reasonState(task, ups.unscaledVector)
                    }
                    if (!task.sequential && task.communicationType !== "Synchronized" && !simulation) {
                        const ctxUpdate = monitorContext(lastSendState.unscaledVector, false)
                        if (ctxUpdate !== undefined && Object.keys(ctxUpdate).length > 0) {
                            for (let el in ctxUpdate) {
                                state.unscaledVector[el] = ctxUpdate[el]
                                state.scaledVector[el] = scaleFeatures(ctxUpdate[el], task.features[el][el].HasRangeStart, task.features[el][el].HasRangeEnd)
                                result.unscaledVector[el] = ctxUpdate[el]
                                result.scaledVector[el] = state.scaledVector[el]
                            }
                        }
                    }

                    const freezedState = state
                    const sendMessage = JSON.stringify(JSON.parse(JSON.stringify({
                        timestamp: new Date(),
                        status: msg.status,
                        state: freezedState,
                        reward: reward,
                        lastState: lastState,
                        lastSendState: lastSendState,
                        lastAction: task.actions[actionQueue[idx]],
                        goal: goal,
                        stateNames: names
                    })))
                    lastAction = actionQueue[idx]
                    client.publish(key, sendMessage, {qos: 2, retain: false})
                    if (task.numActors > 1)
                        idx += 1
                })
                if(!task.sequential) {
                    const upKeys = Object.keys(result.unscaledVector)
                    upKeys.forEach((up) => {
                        lastState.unscaledVector[up] = result.unscaledVector[up]
                        lastState.scaledVector[up] = result.scaledVector[up]
                        lastSendState.unscaledVector[up] = result.unscaledVector[up]
                        lastSendState.scaledVector[up] = result.scaledVector[up]
                    })
                } else {
                    if(reward > 0) {
                        const upKeys = Object.keys(result.unscaledVector)
                        upKeys.forEach((up) => {
                            lastState.unscaledVector[up] = result.unscaledVector[up]
                            lastState.scaledVector[up] = result.scaledVector[up]
                            lastSendState.unscaledVector[up] = result.unscaledVector[up]
                            lastSendState.scaledVector[up] = result.scaledVector[up]
                        })
                    }
                }
            }
        }
        actionQueue.length = 0
        msgCounter = 0

    } else {
            const key = Object.keys(task.topics)[0]
            let names = reasonState(task, lastSendState.unscaledVector)
            const sendMessage = JSON.stringify(JSON.parse(JSON.stringify({
                timestamp: new Date(),
                status: msg.status,
                state: lastSendState,
                reward: -1,
                lastState: lastState,
                lastSendState: lastSendState,
                lastAction: sendAction,
                goal: false,
                stateNames: names
            })))
            client.publish(key, sendMessage, {qos: 2, retain: false})
        }
    })

    })
}

function simulateLocal(task, simulation, topic, message, lastS) {
    const stateRanges = createStateRanges(task)
    let numGoalStates = 0
    for(let s in task.states) {
        if(task.states[s][s].IsGoal) {
            numGoalStates += 1
        }
    }
    let lastAction = undefined
    const featureKeys = Object.keys(task.features)
    let initialState = undefined
    const generateNewState = generateInitialState(task)
    if(lastS === undefined) {
        initialState = generateNewState(false)
    }
    let lastSendState = lastS !== undefined ? JSON.parse(JSON.stringify(lastS.lastSendState)) : JSON.parse(JSON.stringify(initialState))
    let lastState = lastS === undefined ? JSON.parse(JSON.stringify(initialState)) : JSON.parse(JSON.stringify(lastS.lastState))
    let initialStateSize = 0
    for(const s in task.states) {
        if(task.states[s][s].IsInitialState) {
            initialStateSize += 1
        }
    }
    const actionQueue = []
    let lastGoalSize = 0
    const monitorContext = contextControl(task, 5, stateRanges)

    let reasonSR = undefined
    if(task.sequential)
        reasonSR = getSequentialReward(task, stateRanges, numGoalStates)
    else if(task.communicationType == "Synchronized")
        reasonSR = syncReward(task)
    else if(!task.sequential)
        reasonSR = reasonStateReward(task, stateRanges, numGoalStates)
    let msgCounter = 0
    let responses = []
    return async (topic, message, lastS) => {
            if(lastS !== undefined) {
                lastState = JSON.parse(JSON.stringify(lastS.lastState))
                lastSendState = JSON.parse(JSON.stringify(lastS.lastSendState))
                for(const key in lastSendState.unscaledVector) {
                    if(key in lastS.state.unscaledVector) {
                        lastSendState.unscaledVector[key] = lastS.state.unscaledVector[key]
                        lastSendState.scaledVector[key] = lastS.state.scaledVector[key]
                        lastState.unscaledVector[key] = lastS.state.unscaledVector[key]
                        lastState.scaledVector[key] = lastS.state.scaledVector[key]
                    }
                }
            if(lastS.lastAction !== undefined)
                lastAction = Object.keys(lastS.lastAction)[0]
            }
            responses.length = 0
            const msg = JSON.parse(message)
            const step = msg.step
           // console.log(step)
            let sendAction = undefined
            if(msg.action !== undefined)
                sendAction = Object.keys(msg.action)[0]
            if (sendAction in task.actions) {
                let goal = false
                let newGoal = false
                let reward = 0
                if (msg.status === "training" || msg.status === "eval") {
                    const action = msg.action
                    msgCounter += 1
                    const a = Object.keys(msg.action)[0]
                    if (a.length > 0)
                        actionQueue.push(a)
                    if ((task.communicationType == "Synchronized" && msgCounter == task.numActors) || task.communicationType == "Asynchronized") {
                        const lastStates = reasonState(task, lastState.unscaledVector)
                       // console.log(lastState.unscaledVector)
                        let result = undefined
                        const tempState = JSON.parse(JSON.stringify(lastState))

                        const accu = {unscaledVector: {}, scaledVector: {}}
                        for (const actionId of actionQueue) {
                            const updated = updateStateFeaturesToNextState(task, tempState, task.actions[actionId][actionId], stateRanges)
                            for (const f in updated.unscaledVector) {
                                if(f in task.features) {
                                    accu.unscaledVector[f] = updated.unscaledVector[f]
                                    accu.scaledVector[f] = updated.scaledVector[f]
                                }
                            }
                        }

                        result = JSON.parse(JSON.stringify(accu))
                        const tempVec = {unscaledVector: {}, scaledVector: {}}
                        for (const f in lastState.unscaledVector) {
                            if (f in result.unscaledVector) {
                                tempVec.unscaledVector[f] = result.unscaledVector[f]
                                tempVec.scaledVector[f] = result.scaledVector[f]
                            } else {
                                tempVec.unscaledVector[f] = lastState.unscaledVector[f]
                                tempVec.scaledVector[f] = lastState.scaledVector[f]
                            }
                        }
                        if(task.sequential) {
                            const s = reasonState(task, tempVec.unscaledVector)
                            if(lastStates.length > s.length || s.length < 1) {
                                for(const c in lastSendState.unscaledVector) {
                                    tempVec.unscaledVector[c] = lastSendState.unscaledVector[c]
                                    tempVec.scaledVector[c] = lastSendState.scaledVector[c]
                                    result.unscaledVector[c] = lastSendState.unscaledVector[c]
                                    result.scaledVector[c] = lastSendState.scaledVector[c]
                                }
                            }
                        }
                        if (task.communicationType == "Synchronized")
                            reward = reasonSR(tempVec)
                        else
                            reward = reasonSR(lastState, action, lastAction)

                        const resStates = reasonState(task, tempVec.unscaledVector)

                        const cumulativeGoals = resStates.filter((s) => task.states[s][s].IsGoal)

                        const finalState = resStates.filter((s) => task.states[s][s].IsFinalState)

                        if (numGoalStates <= 0 || cumulativeGoals.length >= numGoalStates || (finalState !== undefined && finalState.length > 0)) {
                            goal = true
                        }
                        if (step < 1 || (task.communicationType == "Asynchronized" && goal && numGoalStates > 0) || (task.sequential && lastGoalSize < cumulativeGoals.length)) {
                            const updateContextState = generateNewState(true)
                            for(let r in result.unscaledVector) {
                                if(result.unscaledVector[r] < 0) {
                                    lastState.unscaledVector[r] = 0
                                    lastState.scaledVector[r] = 0
                                    lastSendState.unscaledVector[r] = 0
                                    lastSendState.scaledVector[r] = 0
                                } else {
                                    lastState.unscaledVector[r] = result.unscaledVector[r]
                                    lastState.scaledVector[r] = result.scaledVector[r]
                                    lastSendState.unscaledVector[r] = result.unscaledVector[r]
                                    lastSendState.scaledVector[r] = result.scaledVector[r]
                                }
                            }


                           /* for (const key in lastState.unscaledVector) {
                                if (lastState.unscaledVector[key] < 0) {
                                    lastState.unscaledVector[key] = 0
                                    lastSendState.unscaledVector[key] = 0
                                    lastState.scaledVector[key] = 0
                                    lastSendState.scaledVector[key] = 0
                                }
                            } */
                            result = JSON.parse(JSON.stringify(updateContextState))
                            if (goal)
                                lastGoalSize = 0
                            else
                                lastGoalSize = cumulativeGoals.length
                            if (!task.sequential && !simulation)
                                monitorContext(lastSendState.unscaledVector, true)
                        }


                        const topicKeys = Object.keys(task.topics)
                        let idx = 0

                        topicKeys.forEach(key => {
                            const state = {scaledVector: {}, unscaledVector: {}}
                            const topic = task.topics[key][key]
                            const topicFeatures = Object.keys(topic.HasObservationFeature)
                            const feats = topicFeatures.filter(feat => (feat in result.unscaledVector))
                            feats.forEach((f) => {
                                if (topicFeatures.includes(f)) {
                                    feats.forEach(feat => {
                                        if (result.scaledVector[feat] < 0) {
                                            state.scaledVector[feat] = 0
                                            state.unscaledVector[feat] = 0
                                        } else {
                                            state.scaledVector[feat] = result.scaledVector[feat]
                                            state.unscaledVector[feat] = result.unscaledVector[feat]
                                        }
                                    })
                                }
                            })


                            let names = undefined
                            if (!goal)
                                names = reasonState(task, lastSendState.unscaledVector)
                            else {
                                const ups = {unscaledVector: {}, scaledVector: {}}
                                for (let fe in lastSendState.unscaledVector) {
                                    if (fe in state.unscaledVector)
                                        ups.unscaledVector[fe] = state.unscaledVector[fe]
                                    else
                                        ups.unscaledVector[fe] = lastSendState.unscaledVector[fe]
                                }
                                names = reasonState(task, ups.unscaledVector)
                            }
                            if (!task.sequential && task.communicationType !== "Synchronized" && !simulation) {
                                const ctxUpdate = monitorContext(lastSendState.unscaledVector, false)
                                if (ctxUpdate !== undefined && Object.keys(ctxUpdate).length > 0) {
                                    for (let el in ctxUpdate) {
                                        state.unscaledVector[el] = ctxUpdate[el]
                                        state.scaledVector[el] = scaleFeatures(ctxUpdate[el], task.features[el][el].HasRangeStart, task.features[el][el].HasRangeEnd)
                                        result.unscaledVector[el] = ctxUpdate[el]
                                        result.scaledVector[el] = state.scaledVector[el]
                                    }
                                }
                            }

                            const freezedState = state
                            const sendMessage = JSON.stringify(JSON.parse(JSON.stringify({
                                timestamp: new Date(),
                                status: msg.status,
                                state: freezedState,
                                reward: reward,
                                lastState: lastState,
                                lastSendState: lastSendState,
                                lastAction: task.actions[actionQueue[idx]],
                                goal: goal,
                                stateNames: names
                            })), null, 2)
                            lastAction = actionQueue[idx]
                            //client.publish(key, sendMessage, {qos: 2, retain: false})
                            responses.push(sendMessage)
                            if (task.numActors > 1)
                                idx += 1

                        })

                        const upKeys = Object.keys(result.unscaledVector)
                        upKeys.forEach((up) => {
                            lastState.unscaledVector[up] = result.unscaledVector[up]
                            lastState.scaledVector[up] = result.scaledVector[up]
                            lastSendState.unscaledVector[up] = result.unscaledVector[up]
                            lastSendState.scaledVector[up] = result.scaledVector[up]
                        })
                    }

                }
                const resp = responses[0]
                if(resp !== undefined) {
                    msgCounter = 0
                    actionQueue.length = 0
                }
                return resp

            } else {
                const key = Object.keys(task.topics)[0]
                let names = reasonState(task, lastSendState.unscaledVector)
                const sendMessage = JSON.stringify(JSON.parse(JSON.stringify({
                    timestamp: new Date(),
                    status: msg.status,
                    state: lastSendState,
                    reward: -1,
                    lastState: lastState,
                    lastSendState: lastSendState,
                    lastAction: sendAction,
                    goal: false,
                    stateNames: names
                })))
                //client.publish(key, sendMessage, {qos: 2, retain: false})
                return sendMessage
            }

    }
}

function contextControl(task, maxDuration, stateRanges) {
    const stateMonitor = {}
    for(let s in task.states) {
        stateMonitor[s] = 0
    }
    let negativeActions = []
    for(let na in task.actions) {
        if(task.actions[na][na].IsNegation) {
            negativeActions.push(na)
        }
    }
    const lastS = {}

    return (lastStates, flag) => {
        if (flag) {
            for (const s in stateMonitor) {
                stateMonitor[s] = 0
            }
        } else {
            let acts = []
            let maxStates = []
            const context = reasonState(task, lastStates)
            for (let state of context) {
                if (lastS[state] == lastStates[state])
                    stateMonitor[state] += 1
                if (stateMonitor[state] >= maxDuration) {
                    maxStates.push(state)
                    stateMonitor[state] = 0
                }
            }
            const  vec = {}
            const filteredStates = maxStates.filter((s) => task.states[s][s].IsGoal == false)
            for(let s of filteredStates) {
                const siblings = getSiblingStates(s, stateRanges)
                for(let feat in lastStates) {
                    if(feat in stateRanges[s]) {
                        const negEffects = getFittingActionEffects(task, negativeActions, feat)
                        const newState = siblings.filter((st) => {
                            return (feat in stateRanges[st] && task.states[st][st].IsGoal == false && task.states[st][st].IsInitialState == false
                                && ((stateRanges[st][feat].max < lastStates[feat] && negEffects.includes("DECREASE") || negEffects.includes("OFF"))
                                || (stateRanges[st][feat].min > lastStates[feat] && negEffects.includes("INCREASE") || negEffects.includes("ON"))))
                        })
                        if(newState.length > 0) {
                            const ns = newState[0]
                            const min = stateRanges[ns][feat].min
                            const max = stateRanges[ns][feat].max

                            if(task.features[feat][feat].HasFeatureType == "NUMERIC")
                                vec[feat] = getRandomDouble(min, max)
                            else
                                vec[feat] = getRandomInt(min, max)
                        }
                    }
                }
            }
           /* for (let s of filteredStates) {
                for (let negA of negativeActions) {
                    for (let effect in task.actions[negA][negA].HasEffect) {
                        for (let feat in task.actions[negA][negA].HasEffect[effect].HasObservationFeature) {
                            if (feat in task.states[s][s].HasObservationFeature) {
                                if ((task.actions[negA][negA].HasEffect[effect].HasImpactType == "INCREASE"
                                    && lastStates[feat] <= task.states[s][s].HasObservationFeature[feat].HasRangeStart)
                                    || (task.actions[negA][negA].HasEffect[effect].HasImpactType == "DECREASE"
                                        && lastStates[feat] >= task.states[s][s].HasObservationFeature[feat].HasRangeEnd)
                                    || (task.actions[negA][negA].HasEffect[effect].HasImpactType == "ON"
                                        && lastStates[feat] != 1)
                                    || (task.actions[negA][negA].HasEffect[effect].HasImpactType == "OFF"
                                        && lastStates[feat] != 0)) {
                                    if (!(acts.includes(negA)))
                                        acts.push(negA)
                                }
                            }
                        }
                    }
                }

            } */

            for (let s in lastStates) {
                lastS[s] = lastStates[s]
            }
            return vec
        }
    }
}

function getFittingActionEffects(task, actions, feat) {
    const vec = []
    for(let a of actions) {
        for(let effect in task.actions[a][a].HasEffect) {
            if(feat in task.actions[a][a].HasEffect[effect].HasObservationFeature && !(vec.includes(task.actions[a][a].HasEffect[effect].HasImpactType))) {
                vec.push(task.actions[a][a].HasEffect[effect].HasImpactType)
            }
        }
    } return vec
}
function getSiblingStates(state, stateRanges) {
    const states = []
    const keys = Object.keys(stateRanges[state])
    for(let s in stateRanges) {
        const stateKeys = Object.keys(stateRanges[s])
        if(s !== state) {
            for (let k of keys) {
                if (stateKeys.includes(k) && !(states.includes(s))) {
                    states.push(s)
                }
            }
        }
    }
    return states
}
function reasonStateReward(taskSpec, ranges, goalsSize) {
    const task = JSON.parse(JSON.stringify(taskSpec))
    const stateranges = JSON.parse(JSON.stringify(ranges))
    const numGoals = goalsSize
    let numLastGoals = 0
    //const stateRewardCollection = []
    //const stateMem = []
    return (lastState, action) => {
        let reward = 0
        let actionId = Object.keys(action)[0]
        const updatedStates = updateStateFeaturesToNextState(task, lastState, action[actionId], stateranges)
        const states = reasonState(task, updatedStates.unscaledVector)
        states.forEach((s) => {
            reward += task.states[s][s].HasReward
        })

        /*let stateToReason = {}
        const prevStates = reasonState(task, lastState.unscaledVector)
        const updatedStates = updateStateFeaturesToNextState(task, lastState, action[actionId], stateranges)
        for (let f in lastState.unscaledVector) {
            if (!(f in updatedStates.unscaledVector)) {
                stateToReason[f] = lastState.unscaledVector[f]
            } else {
                stateToReason[f] = updatedStates.unscaledVector[f]
            }
        }
        const states = reasonState(task, stateToReason)
        let effectedStates = states.filter((s) => !(prevStates.includes(s)) && task.states[s][s].IsInitialState == false)
        if (states.length > 0) {
            const goals = states.filter((s) => task.states[s][s].IsGoal == true)
            if(goals.length < numLastGoals) {
                reward = (numLastGoals - goals.length) * (-1 / numGoals)
            } else if(goals.length > numLastGoals || goals.length == numGoals) {
                reward = (goals.length  - numLastGoals) * (1 / numGoals)
            } else {
                stateMem.length = 0
                stateRewardCollection.length = 0
                let effecting = false
                for(let ef in action[actionId].HasEffect) {
                    for(let f in action[actionId].HasEffect[ef].HasObservationFeature) {
                        for (let s of prevStates) {
                            if (f in task.states[s][s].HasObservationFeature && effectedStates.includes(s)) {
                                effecting = true
                                if(!(stateMem.includes(s))) {
                                    stateMem.push(s)
                                    stateRewardCollection.push(task.states[s][s].HasReward)
                                }
                            }
                        }
                    }
                }
                if(effecting) {
                    const stateRew = stateRewardCollection.filter(r => r < 0)
                    if(stateRew.length > 0) {
                        reward = stateRew.reduce((a, b) => a + b)
                        reward *= (1 / numGoals)
                    }
                }
                else
                    reward = -1
            }

            if(goals.length == numGoals)
                numLastGoals = 0
            else
                numLastGoals = goals.length
        } */
        return reward
    }
}

function getSequentialReward(taskSpec, ranges, goalsSize) {
    const task = taskSpec
    const stateranges = ranges
    const numGoals = goalsSize
    //let numActions = 0
    //let lastAction = undefined
    let numLastGoals = 0
    const store = []
    const order = Object.keys(task.states)
    order.pop()
    return (lastState, action, lastAction) => {
        let reward = -0.25
        let actionId = Object.keys(action)[0]
        let stateToReason = {}
        const prevStates = reasonState(task, lastState.unscaledVector)
        const updatedStates = updateStateFeaturesToNextState(task, lastState, action[actionId], stateranges)
        //numActions += 1
        for (let f in lastState.unscaledVector) {
            if (!(f in updatedStates.unscaledVector)) {
                stateToReason[f] = lastState.unscaledVector[f]
            } else {
                stateToReason[f] = updatedStates.unscaledVector[f]
            }
        }
        const states = reasonState(task, stateToReason)
        //let effectedStates = states.filter((s) => !(prevStates.includes(s)) && task.states[s][s].IsInitialState == false)
        if (states.length > 0 && store.length < states.length) {
           const goals = states.filter((s) => task.states[s][s].IsGoal == true)
                /*states.forEach((s) => {
                    reward += task.states[s][s].HasReward
                })*/
            let flag = false
            for(const s of states) {
                if(!(store.includes(s)) && order.indexOf(s) <= store.length) {
                    store.push(s)
                    flag = false
                } else {
                    flag = true
                }
            }

            if(actionId == lastAction || flag) {
                reward = -1
            } else if(goals.length > numLastGoals) {
                reward = goals.length - numLastGoals
                numLastGoals = goals.length
            } else if(numLastGoals > goals.length){
                reward = (numLastGoals - goals.length) * (-1)
                numLastGoals = goals.length
            } else {
                reward = store.length * 0.25
            }
            lastAction = actionId
            if(numLastGoals == numGoals) {
                numLastGoals = 0
                reward = (store.length-1) * 0.25
                store.length = 0
            }
        }
        return reward
    }
}

function syncReward(taskSpec) {
    const task = taskSpec
    return (lastState) => {
        let reward = 0
        const states = reasonState(task, lastState.unscaledVector)
        for(const state of states) {
            reward += task.states[state][state].HasReward
        }
        return reward
    }
}

function updateStateVec(task, vec, action) {
    const updated = {unscaledVector: {}, scaledVector: {}}
    for(let f in vec.unscaledVector) {
        updated.unscaledVector[f] = vec.unscaledVector[f]
        updated.scaledVector[f] = vec.scaledVector[f]
    }
    const effects = action.HasEffect
    for(let effect in effects) {
        const features = effects[effect].HasObservationFeature
        const impactType = effects[effect].HasImpactType
        for (let f in vec.unscaledVector) {
            if (features !== undefined && f in features) {
                if (impactType === "ON") {
                    updated.unscaledVector[f] = 1
                } else if (impactType === "OFF") {
                    updated.unscaledVector[f] = 0
                } else if (impactType === "CONVERT") {
                    vec.unscaledVector[f] == 1 ? updated.unscaledVector[f] = 0 : updated.unscaledVector[f] = 1
                } else if (impactType === "INCREASE") {

                } else if (impactType === "DECREASE") {

                }
            }
        }
    }
    return updated
}

function updateState(task, statevector, action) {
    let featVector = {}
    let scaledFeatVector = {}
    const effects = action.HasEffect
    const effectKeys = Object.keys(effects)
    const stateKeys = Object.keys(statevector.unscaledVector)
    effectKeys.forEach((ek) => {
        if(effects[ek].HasObservationFeature !== null && effects[ek].HasObservationFeature !== undefined) {
            const featKeys = Object.keys(effects[ek].HasObservationFeature)
            const impactType = effects[ek].HasImpactType
            const impactVal = effects[ek].HasImpactRange
            stateKeys.forEach((sk) => {
                if (featKeys.indexOf(sk) !== -1) {
                    const min = task.features[sk][sk].HasRangeStart
                    const max = task.features[sk][sk].HasRangeEnd
                    if(statevector.unscaledVector[sk] < min)  {
                        //const range = max - min
                        //const val = max - (range / 2)
                        statevector.unscaledVector[sk] = min
                        statevector.scaledVector[sk] = scaleFeatures(min, min, max)
                    }
                    if(statevector.unscaledVector[sk] > max) {
                        statevector.unscaledVector[sk] = max
                        statevector.scaledVector[sk] = scaleFeatures(max, min, max)
                    }
                    if (impactType === "INCREASE") {
                        featVector[sk] = statevector.unscaledVector[sk] + impactVal > max ? max : statevector.unscaledVector[sk] + impactVal
                    } else if (impactType === "DECREASE") {
                        featVector[sk] = statevector.unscaledVector[sk] - impactVal < min ? min : statevector.unscaledVector[sk] - impactVal
                    } else if (impactType === "CONVERT") {
                        featVector[sk] = statevector.unscaledVector[sk] === 1 ? 0 : 1
                    } else if (impactType === "COMPUTE") {
                        //TODO: Compute the appropirate function.
                    } else if(impactType === "ON") {
                        featVector[sk] = 1
                    } else if(impactType === "OFF") {
                        featVector[sk] = 0
                    }
                    scaledFeatVector[sk] = scaleFeatures(featVector[sk], min, max)
                }
                /*else if (!(sk in featVector)) {
                    featVector[sk] = statevector.unscaledVector[sk]
                    scaledFeatVector[sk] = statevector.scaledVector[sk]
                }*/
            })
        } else {
            featVector = statevector.unscaledVector
            scaledFeatVector = statevector.scaledVector
        }
    })
    return {scaledVector : scaledFeatVector, unscaledVector : featVector}
}

function getRandomInt(min, max){
    return Math.floor(Math.random() * (parseInt(max) - parseInt(min))) + parseInt(min)
}

function getRandomDouble (min, max) {
    const val = Math.random() * (parseFloat(max) - parseFloat(min)) + parseFloat(min)
    return Math.round(val * 100) / 100
}

async function main(MQTT_HOST) {
    let taskLink = undefined
    process.argv.forEach((val, index) => {
        console.log(`${index} ${val}`)
        if(index === 2) {
           taskLink = val
        }
    })
    let task = undefined
    taskLink = "Go_to_toilet_60"//"Turn_on_light_15" //"PublicGoodsGame" //"HDMTask.jsonld" ////"PublicGoodsGame.jsonld"
    if(!(existsSync(taskLink+"_profile.json")))
        task = getTaskProfile(taskLink+".jsonld")
    else
        task = loadModel(taskLink+"_profile.json")
    const obj = await connectToIoTBroker("ws://localhost:9000", task.agents, task)
    const client = obj.client
    const lastSendState = obj.lastSendState
    const lastState = obj.lastState
    const stateRanges = createStateRanges(task)
    if(task !== null && task !== undefined && task.status === "OPEN") {
        await simulate(task, client,stateRanges, lastSendState, lastState, true)
    }
}

function createStateRanges(task) {
    const stateRanges = {}
    for(let state in task.states) {
        const expr = R.replace(/\)/g, '', R.replace(/\(/g, '', task.states[state][state].HasExpression))
        const tokens = expr.split(" ")
        const toks = tokens.map((t) => {
            const to = t.replace(/\+|\-|\/|\*/g, "_")
            return to
        })
        if(tokens.length > 3) {
            stateRanges[state] = parseComplexExpression(task, toks).ranges
        } else {
            stateRanges[state] = parseExpression(task, toks).ranges
        }

    } return stateRanges
}
//main(/*process.env.MQTT_HOST*/ "ws://localhost:9000")

module.exports = {
    simulateLocal,
    createStateRanges
}