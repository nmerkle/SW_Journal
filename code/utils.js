#!/usr/bin/env node
const {readFileSync, writeFileSync, existsSync} = require('fs')
//const {preprocessRules, oneRule, sequentialCovering} = require('./RuleGenerator.js')
//const {containsObject} = require('./RuleFinder')
const R = require('ramda')

function getTask(path) {
    if(existsSync(path)) {
        const json = readFileSync(path)
        const task = JSON.parse(json)
        /*if(existsSync("./patient.json")) {
            const json = readFileSync("./patient.json")
            const patient = JSON.parse(json)
            return task.concat(patient)
        }*/
        return task
    } else return undefined
}

function scaleFeatures(value, min, max){
    const scalefactor = max - min;
    if(scalefactor !== 0.0) {
        return Math.round(((value - min) / scalefactor) * 100) / 100
    }
    return 0.0
}

function updateStateFeaturesToNextState(task, statevector, action, stateranges) {
    const tempo = {scaledVector: {}, unscaledVector: {}}
    let featVector = {}
    let scaledFeatVector = {}
    const featRanges = {}
    const equations = {}
    const constants = {}
    let statename = reasonState(task, statevector.unscaledVector)
    statename.forEach((sta) => {
        let state = task.states[sta][sta]
        let ruletokens = state.HasExpression.split(" ")
        const toks = ruletokens.map((t) => {
            const to = t.replace(/\+|\-|\/|\*/g, "_").replace(/\(|\)/g, "")
            return to
        })
        let index = 0
        toks.forEach((token) => {
            if (index % 4 === 0) {
                featRanges[token] = {min: undefined, max: undefined}
                if (toks[index + 1] === ">" || toks[index + 1] === ">=") {
                    featRanges[token].min = parseFloat(toks[index + 2])
                } else if (toks[index + 1] === "<" || toks[index + 1] === "<=") {
                    featRanges[token].max = parseFloat(toks[index + 2])
                } else if (toks[index + 1] === "==") {
                    featRanges[token].min = parseFloat(toks[index + 2])
                    featRanges[token].max = parseFloat(toks[index + 2])
                }
            }
            index += 1
        })
        const feats = Object.keys(featRanges)
        feats.forEach((f) => {
            if (featRanges[f].min === undefined) {
                if(f in task.features) {
                    const mini = parseFloat(task.features[f][f].HasRangeStart)
                    featRanges[f].min = mini
                } else {
                    for(let feat in task.features) {
                        if(task.features[feat][feat].HasFeatureType.includes("AGG")) {
                            if(f in task.features[feat][feat].HasObservationFeature) {
                                const mini = parseFloat(task.features[feat][feat].HasObservationFeature[f].HasRangeStart)
                                featRanges[f].min = mini
                            }
                        }
                    }
                }
            }
            if (featRanges[f].max == undefined) {
                if(f in task.features) {
                    const maxi = parseFloat(task.features[f][f].HasRangeEnd)
                    featRanges[f].max = maxi
                } else {
                    for(let feat in task.features) {
                        if(task.features[feat][feat].HasFeatureType.includes("AGG")) {
                            if(f in task.features[feat][feat].HasObservationFeature) {
                                const maxi = parseFloat(task.features[feat][feat].HasObservationFeature[f].HasRangeEnd)
                                featRanges[f].max = maxi
                            }
                        }
                    }
                }
            }
        })
    })
    const effects = action.HasEffect
    const effectKeys = Object.keys(effects)
    const stateKeys = Object.keys(statevector.unscaledVector)
    effectKeys.forEach((ek) => {
        if (effects[ek].HasObservationFeature !== null && effects[ek].HasObservationFeature !== undefined) {
            if (effects[ek].HasImpactRange !== undefined && effects[ek].HasImpactRange !== null) {
                const featKeys = Object.keys(effects[ek].HasObservationFeature)
                const impactType = effects[ek].HasImpactType
                const impactRange = effects[ek].HasImpactRange
                stateKeys.forEach((sk) => {
                    if (featKeys.indexOf(sk) !== -1) {
                        if (sk in task.features) {
                            const min = task.features[sk][sk].HasRangeStart
                            const max = task.features[sk][sk].HasRangeEnd
                            if (min !== undefined && statevector.unscaledVector[sk] < min) {
                                featVector[sk] = min
                                scaledFeatVector[sk] = scaleFeatures(min, min, max)
                            } else if (max !== undefined && statevector.unscaledVector[sk] > max) {
                                featVector[sk] = max
                                scaledFeatVector[sk] = scaleFeatures(max, min, max)
                            } else {
                                let newVal = undefined
                                if (impactType === "DECREASE") {
                                    newVal = statevector.unscaledVector[sk] - impactRange
                                } else if (impactType === "INCREASE") {
                                    newVal = statevector.unscaledVector[sk] + impactRange
                                }
                                featVector[sk] = newVal
                                scaledFeatVector = scaledFeatVector[sk] = scaleFeatures(featVector[sk], min, max)
                            }
                        }
                    }
                })
            } else {
            const featKeys = Object.keys(effects[ek].HasObservationFeature)
            const impactType = effects[ek].HasImpactType
            //const impactVal = effects[ek].HasImpactRange
            stateKeys.forEach((sk) => {
                if (featKeys.indexOf(sk) !== -1) {
                    if (sk in task.features) {
                        const min = task.features[sk][sk].HasRangeStart
                        const max = task.features[sk][sk].HasRangeEnd
                        if (statevector.unscaledVector[sk] < min) {
                            //const range = max - min
                            //const val = max - (range / 2)
                            tempo.unscaledVector[sk] = min
                            tempo.scaledVector[sk] = scaleFeatures(min, min, max)
                        }
                        if (statevector.unscaledVector[sk] > max) {
                            tempo.unscaledVector[sk] = max
                            tempo.scaledVector[sk] = scaleFeatures(max, min, max)
                        }
                        if (sk in featRanges) {
                            const featMin = featRanges[sk].min
                            const featMax = featRanges[sk].max

                            let limit = []
                            for (let s in stateranges) {
                                if (sk in stateranges[s]) {
                                    limit.push(stateranges[s][sk].min)
                                    limit.push(stateranges[s][sk].max)
                                }
                            }
                            limit.sort((a, b) => {
                                return b - a
                            })
                            if (impactType === "INCREASE") {
                                //const impactVal = getRandomInt(featMax, max)
                                let greaterVals = undefined
                                if (featMax < max)
                                    greaterVals = limit.filter(val => val > featMax)
                                else
                                    greaterVals = limit.filter(val => val >= featMax)
                                let impactVal = 0
                                if(greaterVals.length >= 1)
                                    impactVal = getRandomInt(1, Math.abs(greaterVals[greaterVals.length - 1] - featMax))
                                const val = featMax + impactVal
                                featVector[sk] = val > max ? max : val
                            } else if (impactType === "DECREASE") {
                                //const impactVal = getRandomInt(min, featMin)
                                let lowerVals = undefined
                                if (featMin > min)
                                    lowerVals = limit.filter(val => val < featMin)
                                else
                                    lowerVals = limit.filter(val => val <= featMin)
                                const impactVal = getRandomInt(0, Math.abs(featMin - lowerVals[0]))
                                const val = featMin - impactVal
                                featVector[sk] = val < min ? min : val
                            } else if (impactType === "CONVERT") {
                                featVector[sk] = statevector.unscaledVector[sk] === 1 ? 0 : 1
                            } else if (impactType === "ON") {
                                featVector[sk] = 1
                            } else if (impactType === "OFF") {
                                featVector[sk] = 0
                            } else if(impactType === "COMPUTE") {
                                const feat = Object.keys(effects[ek].HasObservationFeature)[0]
                                let evalExpression = ``
                                if(effects[ek].HasParam !== undefined) {
                                    for(const p in effects[ek].HasParam) {
                                        const param = effects[ek].HasParam[p].HasName
                                        const value = effects[ek].HasParam[p].HasValue
                                        evalExpression += `let ${param} = ${value};`
                                    }
                                }
                                evalExpression += `${effects[ek].HasEquation};`
                                equations[feat] = evalExpression
                            }
                            else if(impactType === "CONSTANT") {
                                const constantName = Object.keys(effects[ek].HasObservationFeature)[0]
                                const constant = effects[ek].HasConstant
                                const str = `let ${constantName} = ${constant};`
                                constants[constantName] = str
                                featVector[sk] = constant
                            }
                            scaledFeatVector[sk] = scaleFeatures(featVector[sk], min, max)
                        } else {
                        if (impactType === "CONVERT") {
                                featVector[sk] = statevector.unscaledVector[sk] === 1 ? 0 : 1
                            } else if (impactType === "ON") {
                                featVector[sk] = 1
                            } else if (impactType === "OFF") {
                                featVector[sk] = 0
                            } else if(impactType === "COMPUTE") {
                                const feat = Object.keys(effects[ek].HasObservationFeature)[0]
                                let evalExpression = ``
                                if(effects[ek].HasParam !== undefined) {
                                    for(const p in effects[ek].HasParam) {
                                        const param = effects[ek].HasParam[p].HasName
                                        const value = effects[ek].HasParam[p].HasValue
                                        evalExpression += `let ${param} = ${value};`
                                    }
                                }
                                evalExpression += `${effects[ek].HasEquation};`
                                equations[feat] = evalExpression
                            }
                            else if(impactType === "CONSTANT") {
                                const constantName = Object.keys(effects[ek].HasObservationFeature)[0]
                                const constant = effects[ek].HasConstant
                                const str = `let ${constantName} = ${constant};`
                                constants[constantName] = str
                                featVector[sk] = constant
                            }
                        }
                    } else {
                        for (let f in task.features) {
                            if (task.features[f][f].HasFeatureType.includes("AGG")) {
                                if (sk in task.features[f][f].HasObservationFeature) {
                                    const obs = task.features[f][f].HasObservationFeature[sk]
                                    const min = obs.HasRangeStart
                                    const max = obs.HasRangeEnd
                                    if (statevector.unscaledVector[sk] < min) {
                                        //const range = max - min
                                        //const val = max - (range / 2)
                                        tempo.unscaledVector[sk] = min
                                        tempo.scaledVector[sk] = scaleFeatures(min, min, max)
                                    }
                                    if (statevector.unscaledVector[sk] > max) {
                                        tempo.unscaledVector[sk] = max
                                        tempo.scaledVector[sk] = scaleFeatures(max, min, max)
                                    }
                                    if (sk in featRanges) {
                                        const featMin = featRanges[sk].min
                                        const featMax = featRanges[sk].max
                                        let limit = []
                                        for (let s in stateranges) {
                                            if (sk in stateranges[s]) {
                                                limit.push(stateranges[s][sk].min)
                                                limit.push(stateranges[s][sk].max)
                                            }
                                        }
                                        limit.sort((a, b) => {
                                            return b - a
                                        })
                                        if (impactType === "INCREASE") {
                                            //const impactVal = getRandomInt(featMax, max)
                                            let greaterVals = undefined
                                            if (featMax < max)
                                                greaterVals = limit.filter(val => val > featMax)
                                            else
                                                greaterVals = limit.filter(val => val >= featMax)
                                            let impactVal = 0
                                            if(greaterVals.length >= 1)
                                                impactVal = getRandomInt(1, Math.abs(greaterVals[greaterVals.length - 1] - featMax))
                                            const val = featMax + impactVal
                                            featVector[sk] = val > max ? max : val
                                        } else if (impactType === "DECREASE") {
                                            //const impactVal = getRandomInt(min, featMin)
                                            let lowerVals = undefined
                                            if (featMin > min)
                                                lowerVals = limit.filter(val => val < featMin)
                                            else
                                                lowerVals = limit.filter(val => val <= featMin)
                                            const impactVal = getRandomInt(0, Math.abs(featMin - lowerVals[0]))
                                            const val = featMin - impactVal
                                            featVector[sk] = val < min ? min : val
                                        } else if (impactType === "CONVERT") {
                                            featVector[sk] = statevector.unscaledVector[sk] === 1 ? 0 : 1
                                        } else if (impactType === "ON") {
                                            featVector[sk] = 1
                                        } else if (impactType === "OFF") {
                                            featVector[sk] = 0
                                        } else if(impactType === "COMPUTE") {
                                            const feat = Object.keys(effects[ek].HasObservationFeature)[0]
                                            let evalExpression = ``
                                            if(effects[ek].HasParam !== undefined) {
                                                for(const p in effects[ek].HasParam) {
                                                    const param = effects[ek].HasParam[p].HasName
                                                    const value = effects[ek].HasParam[p].HasValue
                                                    evalExpression += `let ${param} = ${value};`
                                                }
                                            }
                                            evalExpression += `${effects[ek].HasEquation};`
                                            equations[feat] = evalExpression
                                        }
                                        else if(impactType === "CONSTANT") {
                                            const constantName = Object.keys(effects[ek].HasObservationFeature)[0]
                                            const constant = effects[ek].HasConstant
                                            const str = `let ${constantName} = ${constant};`
                                            constants[constantName] = str
                                            featVector[sk] = constant
                                        }
                                        scaledFeatVector[sk] = scaleFeatures(featVector[sk], min, max)
                                    }
                                }
                            }
                        }
                    }
                }
            })
        }
        }
        else {
            featVector = statevector.unscaledVector
            scaledFeatVector = statevector.scaledVector
        }
    })
    effectKeys.forEach((ek) => {
        if(effects[ek].HasImpactType === "COMPUTE") {
            let evalExpression = ``
            const cleaned = effects[ek].HasEquation.replace(/\(|\)|\[|\]/g, "")
            const operands = cleaned.split(/\-|\+|\/|\*|\%/)
            const feat = Object.keys(effects[ek].HasObservationFeature)[0]
            for (const operand of operands) {
                if (operand.trim() in statevector.unscaledVector && !(operand.trim() in constants) && !(evalExpression.includes(operand.trim())))
                    evalExpression += `let ${operand.trim()} = ${statevector.unscaledVector[operand.trim()]};`
            }
            let constantDefinition = ``
            for(const c in constants) {
                if(equations[feat].includes(c))
                    constantDefinition += `${constants[c]}`
            }
            const equation = evalExpression + constantDefinition + equations[feat]
            const result = eval(equation)
            if(result > task.features[feat][feat].HasRangeStart) {
                featVector[feat] = result
               // console.log(`${feat}: ${result}`)
                scaledFeatVector[feat] = scaleFeatures(result, task.features[feat][feat].HasRangeStart, task.features[feat][feat].HasRangeEnd)
            } else {
                featVector[feat] = task.features[feat][feat].HasRangeStart
                scaledFeatVector[feat] = task.features[feat][feat].HasRangeStart
            }

        }
    })

    return {scaledVector : scaledFeatVector, unscaledVector : featVector}
}

function getStringHash(str) {
    let hash = 0
    if(str.length === 0) {
        return hash
    }
    for(let i = 0; i < str.length; i++) {
        let chr = str.charCodeAt(i)
        hash = ((hash << 5) - hash) + chr
        hash = hash & hash
    }
    return Math.abs(hash)
}

function reasonState(task, featureVector) {
  //  console.log(featureVector)
    const reasonedStates = []
    const featKeys = Object.keys(featureVector)
    let paramKeys = undefined
    if(task.params !== undefined)
        paramKeys = Object.keys(task.params)
    const stateKeys = Object.keys(task.states)
    stateKeys.map((val)=>{
        let str = ""
        const expr = task.states[val][val].HasExpression
        //const noBracket = R.replace(/\(|\)/g, '', expr)
        const and = R.replace(/AND/g, '&&', expr)
        const xor = R.replace(/XOR/g, '^', and)
        const ex = R.replace(/OR/g, '||', xor) + ";"
        featKeys.forEach((feat) => {
            if (ex.includes(feat) && feat in featureVector) {
                str += `let ${feat} = ${featureVector[feat]}; `
            }
        })
        if(paramKeys !== undefined) {
            paramKeys.forEach((param) => {
                if (ex.includes(param)) {
                    str += `let ${task.params[param][param].HasName} = ${task.params[param][param].HasValue};`
                }
            })
        }
        if(str !== "") {
            str += ex
            //console.log(str)
            //console.log(expr)
            const result = eval(str)
            if (result)
                reasonedStates.push(val)
        }
    })
    return reasonedStates
}

function reasonReward(task, action, states) {
    let reward = 0.0
    const effects = task.actions[action][action].HasEffect
    const effectKeys = Object.keys(effects)
    let rewardVal = 0
    let punishmentVal = 0
    const isReward = []
    const isPunishment = []
    let evalExpr = ""
    let pEvalExpr = ""
    effectKeys.forEach((key) => {
        const rewardRule = effects[key].HasRewardRule
        const punishmentRule = effects[key].HasPunishmentRule

        if(rewardRule !== undefined) {
            rewardVal = parseFloat(rewardRule.substring(rewardRule.lastIndexOf("=")+1))
            const rewardExpr = rewardRule.substring(0,rewardRule.lastIndexOf("=")-1)
            const rewardExprSplitted = rewardExpr.trim().split(" ")
            const rew = rewardExprSplitted.map((elem) => { if(!elem.includes("AND") && !elem.includes("OR") && !elem.includes("XOR")) return elem += " == 1.0"; else return elem})
            const rewardStates = rewardExpr.trim().split(" ").filter(term => term !== 'OR' && term !== 'AND' && term !== 'XOR' && term !== '=' && term !==rewardRule.split("=")[1])
            const evalReward = rewardStates.reduce(function(acc, val) {
                val =  (states.indexOf(val) !== -1) ? val + " = 1.0; " : val + " = 0.0;";
                return acc + val
            }, "")
            const rEx = rew.reduce((acc, curr) => acc + " " + curr + " ", "")
            const rAND = R.replace(/AND/g, '&&', rEx)
            const rOR = R.replace(/OR/g, '||', rAND)
            const rewEx = R.replace(/XOR/g, '^', rOR)
            evalExpr = evalReward + rewEx+";"

            //if(isReward) reward = (reward + rewardVal) / counter
        }
        if(punishmentRule !== undefined) {
            punishmentVal = parseFloat(punishmentRule.substring(punishmentRule.lastIndexOf("=")+1))
            const punishmentExpr = punishmentRule.substring(0, punishmentRule.lastIndexOf("=")-1)
            const punishmentExprSplitted = punishmentExpr.trim().split(" ")
            const pun = punishmentExprSplitted.map((elem) => {if(!elem.includes("AND") && !elem.includes("OR") && !elem.includes("XOR")) return elem += " == 1.0"; else return elem})
            const punishmentStates = punishmentExpr.trim().split(" ").filter(term => term !== 'OR' && term !== 'AND' && term !== 'XOR' && term !== '=' && term !== punishmentRule.split("=")[1])
            const evalPunishment = punishmentStates.reduce((acc, val) => {
                val = (states.indexOf(val) !== -1) ? val + " = 1.0; " : val + " = 0.0;";
                return acc + val
            }, "")
            const pEx = pun.reduce((acc, curr) => acc + " " + curr + " ", "")
            const pAND = R.replace(/AND/g, '&&', pEx)
            const pOR = R.replace(/OR/g, '||', pAND)
            const punEx = R.replace(/XOR/g, '^', pOR)
            pEvalExpr = evalPunishment + punEx+";"
            //if(isPunishment) reward = (reward + punishmentVal) / counter
        }
        if(eval(evalExpr)) {
            if(isReward.indexOf(evalExpr.trim()) === -1)
                isReward.push(evalExpr.trim())
        }
        if(eval(pEvalExpr)) {
            if (isPunishment.indexOf(pEvalExpr.trim()) === -1)
                isPunishment.push(pEvalExpr.trim())
        }
    })
    if(isReward.length > 0 && isPunishment.length > 0) reward = (rewardVal * isReward.length) + (punishmentVal * isPunishment.length)
    else if(isReward.length > 0) reward = rewardVal
    else if(isPunishment.length > 0) reward = punishmentVal
    return reward
}

function  getAgentProfile(taskPath, nom) {
    const task = getTask(taskPath)
    const ap = task.filter(curr => curr["@type"][0].includes("Agent"))
    let agentIndex = undefined
    for(let ag of ap) {
        if(ag['@id'].includes(nom)) {
            agentIndex = ap.indexOf(ag)
        }
    }
    const id = ap[agentIndex]["@id"].substring(ap[agentIndex]["@id"].lastIndexOf("/")+1)
    const tp = task.filter(curr => curr["@type"][0].includes("Task"))
    const keys = Object.keys(ap[agentIndex])
    const topics = ap[agentIndex][keys.filter(elem => elem.includes("SubscribesFor"))].map(elem => elem["@id"].substring(elem["@id"].lastIndexOf("/") + 1)) //task.filter(curr => curr["@type"][0].includes("Topic"))
    //const topKeys = Object.keys(ap[agentIndex])
    const discount = parseFloat(ap[agentIndex][keys.filter(elem => elem.includes("HasDiscountFactor"))][0]["@value"])
    const epsilon = parseFloat(ap[agentIndex][keys.filter(elem => elem.includes("HasEpsilon"))][0]["@value"])
    const alpha = parseFloat(ap[agentIndex][keys.filter(elem => elem.includes("HasLearningRate"))][0]["@value"])
    const episodes = parseFloat(ap[agentIndex][keys.filter(elem => elem.includes("HasEpisodes"))][0]["@value"])
    const hashing = ap[agentIndex][keys.filter(elem => elem.includes("IsHashing"))][0]["@value"]
    const taskKeys = Object.keys(tp[0])
    //const topics = ap[0][keys.filter(elem => elem.includes("SubscribesFor"))].map(elem => elem["@id"].substring(elem["@id"].lastIndexOf("/")+1))
   /* const topics = topicArray.map((elem) => {
       return  elem[topKeys.filter(elem => elem.includes("HasName"))][0]["@value"]
    })*/
    //const topics = t[topKeys.filter(elem => elem.includes("HasName"))][0]["@value"]
    const actions = ap[agentIndex][taskKeys.filter(elem => elem.includes("HasAction"))].map(elem => elem["@id"].substring(elem["@id"].lastIndexOf("/")+1))
    //const featureSize = tp[0][taskKeys.filter(elem => elem.includes("HasObservationFeature"))].length
    const actionSize = actions.length //ap[0][taskKeys.filter(elem => elem.includes("HasAction"))].length
    const profile = Object.freeze({
        id,
        discount,
        epsilon,
        alpha,
        episodes,
        hashing,
        //featureSize,
        actionSize,
        topics,
        actions
    })
    return profile
}

function getTaskProfile(path) {
    const tsk = getTask(path)
    let task = undefined
    if(tsk !== null && tsk !== undefined) {
        let t = undefined
        if(!(Array.isArray(tsk))) {
            task = tsk.jsonld
        } else
            task = tsk
            t = task.filter((elem) => {
            return elem["@type"][0].includes("Task")
        })
        const keys = Object.keys(t[0])
        const id = t[0]["@id"].substring(t[0]["@id"].lastIndexOf("/") + 1)
        const status = t[0][keys.filter(elem => elem.includes("HasStatus"))].map(elem => elem["@value"])[0]
        const sequential = t[0][keys.filter(elem => elem.includes("IsSequential"))].map(elem => elem["@value"])[0]
        const communicationType = t[0][keys.filter(elem => elem.includes("HasCommunicationType"))].map(elem => elem["@value"])[0]
        const numActors = t[0][keys.filter(elem => elem.includes("HasNumberOfActors"))].map(elem => elem["@value"])[0]
        let paramIds = undefined
        if("http://example.org/Property/Property-3AHasParam" in t[0])
            paramIds = t[0][keys.filter(elem => elem.includes("HasParam"))].map(elem => elem["@id"].substring(elem["@id"].lastIndexOf("/") + 1))
        const actionIds = t[0][keys.filter(elem => elem.includes("HasAction"))].map(elem => elem["@id"].substring(elem["@id"].lastIndexOf("/") + 1))
        const featureIds = t[0][keys.filter(elem => elem.includes("HasObservationFeature"))].map(elem => elem["@id"].substring(elem["@id"].lastIndexOf("/") + 1))
        const stateIds = t[0][keys.filter(elem => elem.includes("HasState"))].map(elem => elem["@id"].substring(elem["@id"].lastIndexOf("/") + 1))
        let ap = undefined
        if(Array.isArray(task))
            ap = task.filter(curr => curr["@type"][0].includes("Agent"))
        else
            ap = task.jsonld.filter(curr => curr["@type"][0].includes("Agent"))
        const agents = []
        for(let aIndex of ap) {
            const agentKeys = Object.keys(aIndex)
            const agent = aIndex["@id"].substring(aIndex["@id"].lastIndexOf("/") + 1)
            agents.push(agent)
        }
        const topicIds = t[0][keys.filter(elem => elem.includes("SubscribesFor"))].map(elem => elem["@id"].substring(elem["@id"].lastIndexOf("/") + 1))
        const topicArray = topicIds.map(elem => buildInstance(elem, task, {[elem]: {}}))
        const topics = array2Object(topicArray)
        const patient = task.filter(curr => curr["@type"][0].includes("Patient"))
        // const patientKeys = Object.keys(patients[0])
        const patientIds = patient.map((elem) => {return elem["@id"].substring(elem["@id"].lastIndexOf("/")+1)})
        const patientArray = patientIds.map((elem) => buildInstance(elem, task, {[elem]: {}}))
        const actionsArray = actionIds.map(elem => buildInstance(elem, task, {[elem]: {}}))
        const actions = array2Object(actionsArray)
        const featuresArray = featureIds.map(elem => buildInstance(elem, task, {[elem]: {}}))
        const features = array2Object(featuresArray)
        const statesArray = stateIds.map(elem => buildInstance(elem, task, {[elem]: {}}))
        const states = array2Object(statesArray)
        let params = undefined
        if(paramIds !== undefined) {
            const paramsArray = paramIds.map(elem => buildInstance(elem, task, {[elem]: {}}))
            params = array2Object(paramsArray)
        }
        //const patientArray = patientIds.map(elem => buildInstance(elem, task, {[elem]: {}}))
        const patients = array2Object(patientArray)
        const profile = Object.freeze({
            id,
            status,
            sequential,
            communicationType,
            numActors,
            params,
            actions,
            features,
            states,
            topics,
            agents,
            patients
        })
        return profile
    } else return undefined
}

function buildInstance(id = undefined, task, object) {
    const newInstance = {}
    const obj = task.filter(elem => (
        id.substring(id.lastIndexOf("/")+1) === elem["@id"].substring(elem["@id"].lastIndexOf("/")+1)
    )/*id.indexOf(elem["@id"].substring(elem["@id"].lastIndexOf("/")+1)) !== -1*/)
    if(obj == undefined || obj == null || obj.length == 0)
        console.log("error")
    const keys = Object.keys(obj[0])
    const filteredKeys = keys.filter(elem => elem.indexOf("@id") === -1 && elem.indexOf("@type") === -1)
    filteredKeys.map(elem => {
        //const property = elem.substring(elem.lastIndexOf("-3A") + 3)
        const property = elem.substring(elem.lastIndexOf("/") + 1)
        const originSubObjects = obj[0][elem].map((o) => ["@id"] in o ? {["@id"]: o["@id"]} : {[o["@type"]]: o["@value"]})
        const subObjects = obj[0][elem].map((o) => ["@id"] in o ? {["@id"]: o["@id"].substring(o["@id"].lastIndexOf("/") + 1)} : {[o["@type"]]: o["@value"]})
        object[id.substring(id.lastIndexOf("/")+1)][property] = {}
        for (let i = 0; i < subObjects.length; i++) {
            const key = Object.keys(subObjects[i])
            const idKey = "@id"
            const name = id.substring(id.lastIndexOf("/")+1)
            idKey in subObjects[i] ? object[name][property][subObjects[i]["@id"]] = {} : key[0].includes("double") ? object[name][property] = parseFloat(subObjects[i][key[0]]) : object[name][property] = subObjects[i][key[0]]
        }
        subObjects.forEach((so)=>{
            const subKeys = Object.keys(so)
            subKeys.forEach((sk)=>{
                if ("@id" in so) {
                    originSubObjects.map(val => {
                        return buildInstance(val["@id"], task, object[id.substring(id.lastIndexOf("/") + 1)][property], object)
                    })
                }
            })
        })


    })
    return object
}

function array2Object(array) {
    const obj= {}
    array.forEach((elem)=>{
        const key = Object.keys(elem)[0]
        obj[key] = elem
    })
    return Object.freeze((obj))
}

function minutesToMilliseconds(minutes) {
    return 1000 * 60 * minutes
}

function gaussianRand() {
    var rand = 0;
    for (var i = 0; i < 6; i += 1) {
        rand += Math.random();
    }
    return rand / 6;
}

function gaussianRandom(start, end) {
    return Math.floor(start + gaussianRand() * (end - start + 1));
}

function getRandomInt(min, max){
    return Math.floor(Math.random() * (parseInt(max) - parseInt(min))) + parseInt(min)
}

function getRandomDouble (min, max) {
    const val = Math.random() * (parseFloat(max) - parseFloat(min)) + parseFloat(min)
    return Math.round(val * 100) / 100
}

function getRandomDecimals(min, max, decimals) {
    const val = Math.random() * (parseFloat(max) - parseFloat(min)) + parseFloat(min)
    return parseFloat(val.toFixed(decimals))
}

function countDecimals(val) {
    if (Math.floor(val.valueOf()) === val.valueOf()) return 0;
    return val.toString().split(".")[1].length || 0;
}
    function precision(tp, fp) {
    const p = tp / (tp + fp)
    return p
}

function recall(tp, fn) {
    const r = tp / (tp + fn)
    return r
}

function f1(precision, recall) {
    const score = 2 * ((precision * recall) / (precision + recall))
    return score
}
function parseComplexExpression(task, tokens) {
    const copiedTokens = tokens.slice()
    const toks = copiedTokens.map((t) => {
        const to = t.replace(/\+|\-|\/|\*/g, "_")
        return to
    })
    let index = 0
    const result = {scaledVector: {}, unscaledVector: {}}
    const ranges = {}
    const operators = {}
    toks.forEach((token) => {
        if(index % 4 === 0) {
            if(index + 2 < toks.length) {
                if(!(token in ranges))
                    ranges[token] = {min: undefined, max: undefined, composition: {and: [], or : [], xor: []}}
                if(index + 4 < toks.length) {
                    if(toks[index +3 ] === "AND" && ranges[token].composition.and.indexOf(toks[index + 4]) === -1)
                        ranges[token].composition.and.push(toks[index + 4])
                    else if(toks[index + 3] === "OR" && ranges[token].composition.or.indexOf(toks[index + 4]) === -1)
                        ranges[token].composition.or.push(toks[index + 4])
                    else if(toks[index + 3] === "XOR" && ranges[token].composition.xor.indexOf(toks[index + 4]) === -1)
                        ranges[token].composition.xor.push(toks[index + 4])
                }
                if(toks[index + 1].indexOf(">") !== -1 || toks[index + 1].indexOf(">=") !== -1) {
                    ranges[token]["min"] = parseFloat(toks[index + 2])
                } else if(toks[index + 1].indexOf("<") !== -1 || toks[index + 1].indexOf("<=") !== -1)
                    ranges[token]["max"] = parseFloat(toks[index + 2])
                else if(toks[index + 1].indexOf("==") !== -1) {
                    ranges[token]["max"] = parseFloat(toks[index + 2])
                    ranges[token]["min"] = parseFloat(toks[index + 2])
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
                    const instance = ranges[cp]
                    let val2 = undefined
                    if(task.features[k][k].HasFeatureType === "NOMINAL") {
                        val2 = getRandomInt(instance.min, instance.max)
                    } else {
                        val2 = getRandomDouble(instance.min, instance.max)
                    }
                    result.unscaledVector[cp] = val2
                    result.scaledVector[cp] = scaleFeatures(val2, instance.min, instance.max)
                })
            }
            if (operand.composition.or.length > 0) {
                operand.composition.or.forEach(cp => {
                    const instance = ranges[cp]
                    let val2 = undefined
                    if(task.features[k][k].HasFeatureType === "NOMINAL") {
                        val2 = getRandomInt(instance.min, instance.max)
                    } else {
                        val2 = getRandomDouble(instance.min, instance.max)
                    }

                    result.unscaledVector[cp] = val2
                    result.scaledVector[cp] = scaleFeatures(val2, instance.min, instance.max)
                })
            }
            if (operand.composition.xor.length > 0) {
                operand.composition.xor.forEach(cp => {
                    const instance = ranges[cp]
                    const range1 = (instance.max * 2) - instance.max
                    const range2 = instance.min
                    let val2 = undefined
                    if(task.features[k][k].HasFeatureType === "NOMINAL") {
                        val2 = [instance.max, instance.min]
                    } else {
                        val2 = [getRandomDouble(instance.max, instance.max * 2), getRandomDouble(0, instance.min)]
                    }
                    const index = getRandomInt(0, 2)
                    result.unscaledVector[cp] = val2[index]
                    index === 0 ? result.scaledVector[cp] = scaleFeatures(val2[index], instance.max, instance.max * 2) : scaleFeatures(val2[index], 0, instance.min)
                })
            }
        }
    })
    return Object.freeze(result)
}

function parseExpression(task, tokens) {
    const copiedTokens = tokens.slice()
    const toks = copiedTokens.map((t) => {
        const to = t.replace(/\+|\-|\/|\*/g, "_")
        return to
    })
    let val = 0
    const result = {unscaledVector: {}, scaledVector: {}}
    const min = task.features[toks[0]][toks[0]].HasRangeStart
    const max = task.features[toks[0]][toks[0]].HasRangeEnd
    if (toks[1] === ">") {
        val = getRandomDouble(parseFloat(toks[2]) + 0.1, max)
    } else if (toks[1] === "<") {
        val = getRandomDouble(min, parseFloat(toks[2]) - 0.1)
    } else if (toks[1] === ">=") {
        val = getRandomDouble(parseFloat(toks[2]), max)
    } else if (toks[1] === "<=") {
        val = getRandomDouble(min, parseFloat(toks[2]))
    } else if (toks[1] === "==") {
        val = parseFloat(toks[2])
    }
    const scaled = scaleFeatures(val, min, max)
    result.unscaledVector[toks[0]] = val
    result.scaledVector[toks[0]] = scaled
    return Object.freeze(result)
}

function deriveActionPattern(task, actionName, agent, isbinned) {
    const ruleExpressions = {}
    let rules = undefined
    let sequentialCoverage = undefined
    if(task.sequential) {
        const {cleanedData, associativeRules, actionsequences} = preprocessRules(`${agent.id}_${agent.algorithm}_${task.id}_N3_rules_original.json`, task.sequential)
        const acts = []
        for (let elem of associativeRules) {
            let rule = {}
            let act = ``
            for (let pred in elem.rule.items) {
                if (pred !== actionName) {
                    //if (elem.rule.items[pred] == 1)
                    rule[pred] = elem.rule.items[pred]
                } else
                    act = `${elem.rule.items[pred]}`
            }
            if (!(act in ruleExpressions))
                ruleExpressions[act] = []
            if (Object.keys(rule).length > 0 && !(containsObject(rule, ruleExpressions[act])))
                ruleExpressions[act].push(rule)
        }
        for (const exp in ruleExpressions) {
            if (ruleExpressions[exp].length <= 0) {
                delete ruleExpressions[exp]
            }
        }
        const derivedRules = {}
        for(const act in ruleExpressions) {
            let len = 0
            let s = undefined
            const states = ruleExpressions[act]
            for(const state of states) {
                const size = Object.keys(state).length
                if(len < size) {
                    len = size
                    s = state
                }
            }
            if(!(act in derivedRules))
                derivedRules[act] = {default: false, rules: []}
            let expr = ``
            for(const feat in s) {
                if(s[feat] == 1)
                    expr += `${feat} == true && `
                else
                    expr += `${feat} == false && `
            }
            expr = expr.substring(0, expr.lastIndexOf("&&")).trim()
            derivedRules[act].rules.push({rule: expr})
        }
        return {derivedRules, actionsequences, rules, sequentialCoverage}
    } else {
        let dataset = undefined
        let datasetNumeric = undefined
        let datasetNominal = undefined
        let datasetAll = undefined
        if(!isbinned)
            dataset = `${agent.id}_${agent.algorithm}_${task.id}_N3_rules.json`
        else {
            if(existsSync(`${agent.id}_${agent.algorithm}_${task.id}_N3_rules_binned_numeric.json`)) {
                datasetNumeric = `${agent.id}_${agent.algorithm}_${task.id}_N3_rules_binned_numeric.json`
            }
            if(existsSync(`${agent.id}_${agent.algorithm}_${task.id}_N3_rules_binned_nominal.json`)) {
                datasetNominal = `${agent.id}_${agent.algorithm}_${task.id}_N3_rules_binned_nominal.json`
            }
            if(existsSync(`${agent.id}_${agent.algorithm}_${task.id}_N3_rules_binned_all.json`)) {
                datasetAll = `${agent.id}_${agent.algorithm}_${task.id}_N3_rules_binned_all.json`
            }
        }
        const featKeys = Object.keys(task.features)
        const feats2Bin = featKeys.filter((f) => task.features[f][f].HasFeatureType === "NUMERIC" && f !== 'ActionImpact')
        const featsNumeric = featKeys.filter((f) => task.features[f][f].HasFeatureType === "NOMINAL")
        if (datasetNumeric !== undefined && feats2Bin !== undefined && feats2Bin.length > 0 && featsNumeric === undefined || featsNumeric.length <= 0) {
            dataset = datasetNumeric
        } else if (datasetNominal !== undefined && feats2Bin !== undefined && feats2Bin.length <= 0 && featsNumeric !== undefined && featsNumeric.length > 0) {
            dataset = datasetNominal
        } else if (datasetAll !== undefined && feats2Bin !== undefined && feats2Bin.length > 0 && featsNumeric !== undefined && featsNumeric.length > 0) {
            dataset = datasetAll
        }
        const {cleanedData, frequentItems, associativeRules} = preprocessRules(dataset, task.sequential, feats2Bin)
        let concatenated = undefined
        if(isbinned) {
            const filteredData = cleanedData.map((d) => {
                for(const o in d) {
                    if(d[o] === 0)
                        d[o] = false
                    else if(d[o] === 1)
                        d[o] = true
                }
                return Object.freeze(d)
            })
            if (feats2Bin !== undefined && feats2Bin.length > 0 && featsNumeric !== undefined && featsNumeric.length > 0) {
                concatenated = feats2Bin.concat(featsNumeric)
                rules = oneRule(Object.keys(task.actions), concatenated, filteredData, "Action", task)
                sequentialCoverage = sequentialCovering(task, Object.keys(task.actions), concatenated, filteredData, "Action", isbinned)
            } else if (feats2Bin !== undefined && feats2Bin.length > 0 && featsNumeric === undefined || featsNumeric.length <= 0) {
                sequentialCoverage = sequentialCovering(task, Object.keys(task.actions), feats2Bin, filteredData, "Action", isbinned)
                rules = oneRule(Object.keys(task.actions), feats2Bin, filteredData, "Action", task)
            } else if (feats2Bin !== undefined && feats2Bin.length <= 0 && featsNumeric !== undefined && featsNumeric.length > 0) {
                rules = oneRule(Object.keys(task.actions), featsNumeric, filteredData, "Action", task)
                sequentialCoverage = sequentialCovering(task, Object.keys(task.actions), featsNumeric, filteredData, "Action", isbinned)
            }
        } else {
            if (feats2Bin !== undefined && feats2Bin.length > 0 /*&& featsNumeric.length <= 0*/)
                sequentialCoverage = sequentialCovering(task, Object.keys(task.actions), feats2Bin, cleanedData, "Action", isbinned)
            else if (feats2Bin !== undefined && feats2Bin.length <= 0 && featsNumeric !== undefined && featsNumeric.length > 0)
                sequentialCoverage = sequentialCovering(task, Object.keys(task.actions), featsNumeric, cleanedData, "Action", isbinned)
            else if (feats2Bin !== undefined && feats2Bin.length > 0 && featsNumeric !== undefined && featsNumeric.length > 0)
                sequentialCoverage = sequentialCovering(task, Object.keys(task.actions), concatenated, cleanedData, "Action", isbinned)
        }
    }
    return {ruleExpressions, rules, sequentialCoverage}
}

function binData(task, store, feats2bin, binFactor) {
    const binnedData = {}
    const result = []
    for(const feat of feats2bin) {
        const start = task.features[feat][feat].HasRangeStart
        const end = task.features[feat][feat].HasRangeEnd
        const range = end - start
        const binRange = end * binFactor
        let diff = end
        while (diff > 0) {
            if(!(feat in binnedData))
                binnedData[feat] = []
            if(diff == task.features[feat][feat].HasRangeEnd) {
                binnedData[feat].push({key: `${feat}_${diff - binRange}_${diff}`, start: diff - binRange, end: diff})
            } else
                binnedData[feat].push({key: `${feat}_${diff - binRange}_${diff-1}`,  start: diff - binRange, end: diff-1})
            diff -= binRange
        }
    }
    for(const r of store) {
        const rules = []
        for(const row in r.rules) {
            const state = {}
            for(const b in binnedData) {
                for(const key of binnedData[b])
                    state[key.key] = undefined
            }
            let action = r.rules[row].action
            for(const f in r.rules[row].state) {
                if(f in binnedData) {
                    const val = r.rules[row].state[f]
                    for (const bin of binnedData[f]) {
                        if (val >= bin.start && val <= bin.end) {
                            state[bin.key] = 1
                        } else {
                            state[bin.key] = 0
                        }
                    }
                }
            }
            rules.push({state: state, action: action})
            const rule = {rules: rules, reward: r.reward}
            result.push(rule)
        }
    }
    return result
}

module.exports = {
    getRandomInt,
    getRandomDouble,
    getRandomDecimals,
    getTaskProfile,
    getAgentProfile,
    minutesToMilliseconds,
    reasonState,
    reasonReward,
    gaussianRandom,
    scaleFeatures,
    precision,
    recall,
    f1,
    binData,
    updateStateFeaturesToNextState,
    getStringHash,
    parseExpression,
    parseComplexExpression,
    countDecimals,
    deriveActionPattern
}