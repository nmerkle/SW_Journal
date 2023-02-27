#! usr/bin/env node

const {readdirSync, readFileSync, renameSync, writeFileSync, createReadStream, existsSync, mkdirSync} = require('fs')
const readline = require('readline')
const stream = require('stream')
const {getRandomInt, getTaskProfile, reasonState} = require('./utils')
const {simulateLocal} = require('./simulator')
const {v4: uuidv4} = require("uuid");

function readLines({ input }) {
    const output = new stream.PassThrough({ objectMode: true });
    const rl = readline.createInterface({ input });
    rl.on("line", line => {
        output.write(line);
    });
    rl.on("close", () => {
        output.push(null);
    });
    return output;
}

async function preprocessFiles() {
    const dir = 'C:/Users/nirpsel/Downloads/programs_processed_precond_nograb_morepreconds/withoutconds/results_intentions_march-13-18'
    const samples = {}
    const dataSet = []
    const taskCounter = {}
    const files = readdirSync(dir)

    for (const file of files) {
        const path = `${dir}/${file}`
        const input = createReadStream(path);
        let lineNumber = 0
        let task = undefined
        for await (const line of readLines({input})) {
            console.log(line)
            if(lineNumber == 0) {
                task = file.substring(0, file.lastIndexOf(".")).trim() //line
                if (!(task in samples)) {
                    samples[task] = []
                    taskCounter[task] = -1
                }
                samples[task].push([])
                taskCounter[task] += 1
            }
            else if(line.includes("[") || line.includes("]")) {
                const subs = line.substring(line.indexOf("<"))
                const splits = subs.split("<")
                let con = ""
                let len = splits.length
                for(const sp of splits) {
                    if(sp !== "") {
                        const term = sp.substring(0, sp.indexOf('>')) //+ " "
                        con += term
                        const val = sp.substring(sp.indexOf('(') + 1, sp.indexOf(')'))
                        if(val !== "")
                            con += `_${val}`
                        if(len > 2) {
                            len = 0
                            con += '_To_'
                        }
                    }
                }
                let action = undefined
                if(con !== "")
                    action = `${line.substring(line.indexOf('[')+1, line.indexOf(']'))} ${con}`.trim().replace(/ /g, "_")
                else
                    action = `${line.substring(line.indexOf('[')+1, line.indexOf(']'))}`.trim().replace(/ /g, "_")
                samples[task][taskCounter[task]].push(action)
            }
            lineNumber += 1
        }
    }
    const taskIndexes = {}
    const taskIndexArray = []
    const stateIndexes = {}
    const stateIndexArray = []
    let num = 0
    let sNum = 0
    const len = Object.keys(samples).length
    for(let i = 0; i < len; i++) {
        taskIndexArray.push(undefined)
    }

    for(const key in samples) {
        const state = []
        for(const a of samples[key][0]) {
            state.push(a+"_Done")
            if(!(a+"_Done" in stateIndexes)) {
                stateIndexes[a+"_Done"] = sNum
                stateIndexArray.push(a+"_Done")
                sNum += 1
            }
        }
        dataSet.push([key, samples[key], [state]])
        taskIndexes[key] = num
        taskIndexArray.splice(num, 1, key)
        num++
    }
    return {taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, dataSet}
}

function semantifyAndStoreAgent(agent, task) {
    const entity = 'http://example.org/Entity/'
    const prop = 'http://example.org/Property/Property-3A'
    const concept = 'http://example.org/Concept/Category-3A'
    const xmlSchema = 'http://www.w3.org/2001/XMLSchema#'
    const agentEntity = {
        '@id': entity + agent.id,
        '@type': [concept + 'Agent'],
        [prop + 'HasName']: [{'@value': agent.id}],
        [prop + 'HasAlgorithm']: [{'@value': agent.algorithm}],
        [prop + 'HasLearningRate']: [{'@type': xmlSchema+'double', '@value': parseFloat(agent.alpha)}],
        [prop + 'HasEpsilon']: [{'@type': xmlSchema+'double', '@value': parseFloat(agent.greedy)}],
        [prop + 'HasDiscountFactor']: [{'@type': xmlSchema+'double', '@value': parseFloat(agent.epsilon)}],
        [prop + 'HasEpisodes']: [{'@type': xmlSchema+'integer', '@value': parseFloat(agent.episodes)}],
        [prop + 'IsHashing']: [{'@type': xmlSchema+'boolean', '@value': agent.hashing}],
        [prop + 'SubscribesFor']: [],
        [prop + 'HasTask']: [{'@id': task.id}],
        [prop + 'HasAction']: []
    }
    task.topics.forEach((topic) => {
        agentEntity[prop + 'SubscribesFor'].push({'@id': topic.id})
    })
    task.actions.forEach((action) => {
        agentEntity[prop + 'HasAction'].push({'@id': action.id})
    })
    return agentEntity

}

function createTaskDescriptions(originTask, taskIndexes, dataset) {
    let semT1 = undefined
    let t1Json = undefined
    let dir = '../VirtualHomeKG/JSONLD/';
    let dir2 = '../VirtualHomeKG/TURTLE/';

    if (!existsSync(dir)){
        mkdirSync(dir, { recursive: true });
    }
    if (!existsSync(dir2)){
        mkdirSync(dir2);
    }
    let rootDir = `VirtualHomeKG`
    const topicId = `TaskTopic`
    const topic = {id: topicId, features: []}
    let featTracker = []
    const initState = {
        id: 'InitialState_'+originTask,
        expression: '',
        actions: [],
        features: [],
        goal: false,
        initial: true,
        reward: 0
    }
    const finalState = {
        id: 'FinalState_'+originTask,
        expression: '',
        actions: [],
        features: [],
        goal: true,
        initial: false,
        reward: 0
    }
    const index = taskIndexes[originTask]
    const t = dataset[index]
    const t1 = {
        id: t[0],
        sequential: true,
        communicationType: 'Asynchronized',
        numberOfActors: 1,
        features: [],
        states: [],
        actions: [],
        topics: [],
        effects: {}
    }
    //for(const t of dataset) {
        featTracker.length = 0
        let count = 0
        const initState1 = {
            id: 'InitialState_'+originTask,
            expression: '',
            actions: [],
            features: [],
            goal: false,
            initial: true,
            reward: 0
        }
        const topic1ID = `TaskTopic`
        const topic1 = {id: topic1ID, features: []}
        //let i = 0
        t[1][0].forEach((act) => {
            const stateId = `${act}_Done`
            const featId = `Is${act}`
            featTracker.push(featId)
            const actionId = `${act}`
            const effectId = `Set${act}`
            let g = false
            let rew = 0
            let exp = ``
           /* let prevState = undefined
            if(i > 0 && i < t[1][0].length)
                prevState = t[1][0][i-1]+"Done"
            else if(i === 0)
                prevState = initState.id
            else
                prevState = finalState.id
            const transition = {
                id:  uuidv4(),
                action: act,
                previousState: prevState,
                probability: 1.0,
                nextState: stateId
            }
            i += 1 */
            const state = {
                id: stateId,
                expression: ``,
                actions: [actionId],
                features: [],
                goal: false,
                initial: false,
                reward: 0
            }
            featTracker.forEach((f) => {
                exp += `${f} == 1 AND `
                state.features.push(f)
            })
            exp = exp.substring(0, exp.lastIndexOf("AND")).trim()
            state.expression = exp
            if (count === t[1][0].length - 1) {
                g = false//true
                finalState.expression = exp
                finalState.actions.push(act)
                let initExp = ``
                featTracker.forEach((f) => {
                    initExp += `${f} == 0 AND `
                    finalState.features.push(f)
                })
                initExp = initExp.substring(0, initExp.lastIndexOf("AND")).trim()
                initState1.expression += initExp
                rew = 1
                featTracker.length = 0
            }

            state.goal = g
            state.initial = false
            state.reward = rew
            initState1.actions.push(actionId)
            initState1.features.push(featId)
            const action = {id: `${actionId}`, effects: [effectId], /*transitions: [transition],*/ negation: false, text: ""}
            const feat = {id: featId, rangeStart: 0, rangeEnd: 1, type: 'NOMINAL', unit: ""}
            const effect = {id: effectId, impactType: "ON", features: [featId]}
            topic1.features.push(feat.id)
            t1.states.push(state)
            t1.actions.push(action)
            t1.features.push(feat)
            t1.effects[effectId] = effect
            count += 1
        })
        t1.topics.push(topic1)
        t1.states.push(initState1)
        t1.states.push(finalState)
        semT1 = semantifyAndStoreTask(t1)
        t1Json = JSON.stringify(semT1.jsonld, null, 2)
        writeFileSync(`../${rootDir}/JSONLD/${t[0]}.jsonld`, t1Json)
        writeFileSync(`../${rootDir}/TURTLE/${t[0]}.ttl`, semT1.turtle)
        //featTracker.length = 0
        //count = 0
   // }
    return {task: t1, jsonld: t1Json, ttl: semT1.turtle}
}

function semantifyAndStoreTask(task) {
    let turtle = ``
    const jsonld = []
    const entity = 'http://example.org/Entity/'
    const prop = 'http://example.org/Property/'
    const concept = 'http://example.org/Concept/'
    const xmlSchema = 'http://www.w3.org/2001/XMLSchema#'
    const agent = {id: `TaskEmbeddingAgent`, alpha: 0.01, epsilon: 0.7, greedy: 0.2, algorithm: "dqn", episodes: 20000, hashing: false}
    const agentInstance = semantifyAndStoreAgent(agent, task)
    let taskEntity = {
        '@id': entity+task.id,
        '@type': [concept+'Task'],
        [prop+'HasStatus']: [{'@value':"OPEN"}],
        [prop+'IsSequential']: [{'@type': xmlSchema+'boolean', '@value': task.sequential}],
        [prop+'HasNumberOfActors']: [{'@type': xmlSchema+'integer', '@value': task.numberOfActors}],
        [prop+'HasCommunicationType']: [{'@type': xmlSchema+'string', '@value': task.communicationType}],
        [prop+'HasState']: [],
        [prop+'HasAction']: [],
        [prop+'HasObservationFeature']: [],
        [prop+'SubscribesFor']: []
    }
    turtle += `@prefix entity: <${entity}> .\n`
    turtle += `@prefix property: <${prop}> .\n`
    turtle += `@prefix concept: <${concept}> .\n`
    turtle += `@prefix xsd: <${xmlSchema}> .\n`
    turtle += `@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n`
    turtle += `@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n`

    turtle += `entity:${task.id} a concept:Task;\n`
    turtle += `\tproperty:HasStatus "OPEN"^^xsd:string;\n`
    turtle += `\tproperty:IsSequential "${task.sequential}"^^xsd:boolean;\n`
    turtle += `\tproperty:HasNumberOfActors "${task.numberOfActors}"^^xsd:integer;\n`
    turtle += `\tproperty:HasCommunicationType "${task.communicationType}"^^xsd:string;\n`
    task.states.forEach((state) => {
        let id = undefined
        if(state.id.includes("-") || state.id.includes("+") || state.id.includes("/") || state.id.includes("*")) {
            id = state.id.replace(/\-|\+|\\|\*/g, "_")
        } else {
            id = state.id
        }
        turtle += `\tproperty:HasState entity:${id};\n`
    })
    task.features.forEach(feat => {
        let id = undefined
        if(feat.id.includes("-") || feat.id.includes("+") || feat.id.includes("/") || feat.id.includes("*")) {
            id = feat.id.replace(/\-|\+|\\|\*/g, "_")
        } else {
            id = feat.id
        }
        turtle += `\tproperty:HasObservationFeature entity:${id};\n`
    })
    task.actions.forEach(action => {
        turtle += `\tproperty:HasAction entity:${action.id};\n`
    })
    task.topics.forEach(topic => {
        turtle += `\tproperty:SubscribesFor entity:${topic.id};\n`
    })
    turtle = turtle.substring(0, turtle.lastIndexOf(";"))
    turtle += ` .\n`
    task.states.forEach((state) => {
        let id = undefined
        if(state.id.includes("-") || state.id.includes("+") || state.id.includes("/") || state.id.includes("*")) {
            id = state.id.replace(/\-|\+|\\|\*/g, "_")
        } else {
            id = state.id
        }
        const isFinal = (state.goal == true) ? true : false
        let st = {'@id': entity+id, '@type': [concept+'State'],
            [prop+'HasObservationFeature']: [],
            [prop+'HasAction']: [],
            [prop+'IsGoal']: [{'@type': xmlSchema+'boolean', '@value': state.goal}],
            [prop+'IsFinalState']: [{'@type': xmlSchema+'boolean', '@value': isFinal}],
            [prop+'IsInitialState']: [{'@type': xmlSchema+'boolean', '@value': state.initial}],
            [prop+'HasExpression']: [{'@value':state.expression}],
            [prop+'HasReward']: [{'@type': xmlSchema+'double', '@value': state.reward}]
        }
        turtle += `entity:${id} a concept:State;\n`
        turtle += `\tproperty:IsGoal "${state.goal}"^^xsd:boolean;\n`
        turtle += `\tproperty:IsFinalState "${isFinal}"^^xsd:boolean;\n`
        turtle += `\tproperty:IsInitialState "${state.initial}"^^xsd:boolean;\n`
        turtle += `\tproperty:HasExpression "${state.expression}"^^xsd:string;\n`
        turtle += `\tproperty:HasReward "${state.reward}"^^xsd:double;\n`
        state.features.forEach((feat) => {
            let elem = {'@id': undefined}
            let fId = undefined
            if(feat.includes("-") || feat.includes("+") || feat.includes("/") || feat.includes("*")) {
                fId = feat.replace(/\-|\+|\\|\*/g, "_")
            } else {
                fId = feat
            }

            elem['@id'] = entity+fId
            st[prop+'HasObservationFeature'].push(elem)
            turtle += `\tproperty:HasObservationFeature entity:${fId};\n`
        })

        state.actions.forEach((act) => {
            let elem = {'@id': undefined}
            elem['@id'] = entity+act
            st[prop+'HasAction'].push(elem)
            turtle += `\tproperty:HasAction entity:${act};\n`
        })
        turtle = turtle.substring(0, turtle.lastIndexOf(";"))
        turtle += ` .\n`
        jsonld.push(st)
        taskEntity[prop+'HasState'].push({'@id': st["@id"]})
    })

    task.features.forEach((feat) => {
        let id = undefined
        if(feat.id.includes("-") || feat.id.includes("+") || feat.id.includes("/") || feat.id.includes("*")) {
            id = feat.id.replace(/\-|\+|\\|\*/g, "_")
        } else {
            id = feat.id
        }
        let feature = {
            '@id': entity+id,
            '@type': [concept+'ObservationFeature'],
            [prop+'HasRangeStart']: [{'@type': xmlSchema+'double', '@value':feat.rangeStart}],
            [prop+'HasRangeEnd']: [{'@type': xmlSchema+'double', '@value':feat.rangeEnd}],
            [prop+'HasFeatureType']: [{'@value':feat.type}],
            [prop+'HasUnit']: [{'@value': feat.unit}]
        }
        turtle += `entity:${id} a concept:ObservationFeature;\n`
        turtle += `\tproperty:HasRangeStart "${feat.rangeStart}"^^xsd:double;\n`
        turtle += `\tproperty:HasRangeEnd "${feat.rangeEnd}"^^xsd:double;\n`
        turtle += `\tproperty:HasFeatureType "${feat.type}"^^xsd:string;\n`
        turtle += `\tproperty:HasUnit "${feat.unit}"^^xsd:string .\n`
        jsonld.push(feature)
        taskEntity[prop+'HasObservationFeature'].push({'@id': feature["@id"]})
    })

    task.topics.forEach((topic) => {
        let top = {
            '@id': entity+topic.id,
            '@type': [concept+'Topic'],
            [prop + 'HasName']: [{'@value': topic.id}],
            [prop+'HasObservationFeature']: []
        }
        turtle += `entity:${topic.id} a concept:Topic;\n`
        turtle += `\tproperty:HasName "${topic.id}"^^xsd:string;\n`
        topic.features.forEach((feat) => {
            let fId = undefined
            if(feat.includes("-") || feat.includes("+") || feat.includes("/") || feat.includes("*")) {
                fId = feat.replace(/\-|\+|\\|\*/g, "_")
            } else {
                fId = feat
            }
            let feature = {
                '@id': entity+fId
            }
            top[prop+'HasObservationFeature'].push(feature)
            turtle += `\tproperty:HasObservationFeature entity:${fId};\n`
        })
        turtle = turtle.substring(0, turtle.lastIndexOf(";"))
        turtle += ` .\n`
        jsonld.push(top)
        taskEntity[prop+'SubscribesFor'].push({'@id': top["@id"]})
    })

    for(let effect in task.effects) {
        let ef = {
            '@id': entity+effect,
            '@type': [concept+'Effect'],
            [prop+'HasImpactType']: [{'@value': task.effects[effect].impactType}],
            [prop+'HasObservationFeature']: []
        }
        turtle += `entity:${effect} a concept:Effect;\n`
        turtle += `\tproperty:HasImpactType "${task.effects[effect].impactType}"^^xsd:string;\n`
        task.effects[effect].features.forEach((feat) => {
            let fId = undefined
            if(feat.includes("-") || feat.includes("+") || feat.includes("/") || feat.includes("*")) {
                fId = feat.replace(/\-|\+|\\|\*/g, "_")
            } else {
                fId = feat
            }
            let feature = {
                '@id': entity+fId
            }
            turtle += `\tproperty:HasObservationFeature entity:${fId};\n`
            ef[prop+'HasObservationFeature'].push(feature)

        })
        turtle = turtle.substring(0, turtle.lastIndexOf(";"))
        turtle += ` .\n`
        jsonld.push(ef)
    }

    task.actions.forEach((action) => {
       /* const transId = action.transition.id
        const nextState = action.transition.nextState
        const previouseState = action.transition.previousState
        const prob = action.transition.probability
        let trans = {
            '@id': entity+transId,
            '@type': [concept+'Transition'],
            [prop+'HasPreviousState']: [{'@id': `${previouseState}`}],
            [prop+'HasNextState']: [{'@id': `${nextState}`}],
            [prop+'HasAction']: [{'@id': `${action.transition.action}`}],
            [prop+'HasTransitionProbability']: [
                {
                '@type': `${xmlSchema}double`,
                '@value': prob
                }
            ]
        }
        turtle += `entity:${transId} a concept:Transition;\n`
        turtle += `\tproperty:HasPreviousState entity:${previouseState};\n`
        turtle += `\tproperty:HasNextState entity:${nextState};\n`
        turtle += `\tproperty:HasAction entity:${action.transition.action};\n`
        turtle += `\tproperty:HasTransitionProbability "${prob}"^^xsd:double.\n`

        jsonld.push(trans) */
        let act = {
            '@id': entity+action.id,
            '@type': [concept+'Action'],
            [prop+'HasText']: [{'@value': action.text}],
            //[prop+'IsNegation']: [{'@type': xmlSchema+'boolean', '@value': action.negation}],
            [prop+'HasEffect']: []
           // [prop+ 'HasTransition']: [],
        }
       // act[prop+'HasTransition'].push(`{'@id': ${entity}${transId}}`)
        turtle += `entity:${action.id} a concept:Action;\n`
        turtle += `\tproperty:HasText "${action.text}"^^xsd:string;\n`
       // turtle += `\tproperty:HasTransition entity:${transId};\n`
        //turtle += `\tproperty:IsNegation "${action.negation}"^^xsd:boolean;\n`
        action.effects.forEach((effect) => {
            let ef = {
                '@id': entity+effect
            }
            act[prop+'HasEffect'].push({'@id': ef["@id"]})
            turtle += `\tproperty:HasEffect entity:${effect};\n`
        })
        turtle = turtle.substring(0, turtle.lastIndexOf(";"))
        turtle += ` .\n`
        jsonld.push(act)
        taskEntity[prop+'HasAction'].push({'@id': act["@id"]})
    })
    jsonld.push(taskEntity)
    jsonld.push(agentInstance)
    return {jsonld: jsonld, turtle: turtle}
}

function getEmbeddings() {
    const taskEmbeddings = {}
    const taskLines = readFileSync("./taskEmbeddingsMerged.tsv").toString().split("\n")
    const taskLabels = readFileSync("./taskMetaTSVMerged.tsv").toString().split("\n")
    let index = 0
    for(const label of taskLabels) {
        taskEmbeddings[label] = taskLines[index].split("\t").map(a => parseFloat(a))
        index += 1
    }
    const stateEmbeddings = {}
    index = 0
    const stateLines = readFileSync("./stateEmbeddingsMerged.tsv").toString().split("\n")
    const stateLabels = readFileSync("./stateMetaTSVMerged.tsv").toString().split("\n")
    for(const label of stateLabels) {
        stateEmbeddings[label] = stateLines[index].split("\t").map(a => parseFloat(a))
        index += 1
    }

    const actionEmbeddings = {}
    index = 0
    const actionLines = readFileSync("./actionEmbeddingsMerged.tsv").toString().split("\n")
    const actionLabels = readFileSync("./actionMetaTSVMerged.tsv").toString().split("\n")
    for(const label of actionLabels) {
        actionEmbeddings[label] = actionLines[index].split("\t").map(a => parseFloat(a))
        index += 1
    }
    return {taskEmbeddings, stateEmbeddings, actionEmbeddings}
}

function cosineSimilarity(v1, v2) {
    if(v1 !== undefined && v2 !== undefined && v1.length == v2.length) {
        let lenV1 = 0
        let lenV2 = 0
        let nominator = 0
        for(let i = 0; i < v1.length; i++) {
            nominator += v1[i] * v2[i]
            lenV1 += Math.pow(v1[i], 2.0)
            lenV2 += Math.pow(v2[i], 2.0)
        }
        const norm1 = Math.sqrt(lenV1)
        const norm2 = Math.sqrt(lenV2)
        const similarity = nominator / (norm1 * norm2)
        return similarity
    }
    return -1
}

async function main(numTasks, numRuns, distLimit) {
    /*const loadedCSV = readFileSync('./AgentEnsemblesEval.csv').toString()
    const splittedCSV = loadedCSV.split("\n")
    const row = splittedCSV.map(a => a.split(','))
    const samples = []
    let copiedCSV = ``
    for(const r of row) {
        if(!(samples.includes(r[0]))) {
            samples.push(r[0])
            for(const elem of r) {
                copiedCSV += elem+','
            }
            copiedCSV = copiedCSV.substring(0, copiedCSV.lastIndexOf(',')).trim()
            copiedCSV += '\n'
        }
    }
    writeFileSync('./AgentEnsemblesEval2.csv', copiedCSV)
    console.log(loadedCSV)*/

    let csv = `Task,CumulativeReward,NumWrongDecisions,Steps,MaxDistance,NumPolicies,Episodes\n`
    const actionDistribution = {}
    const composedPolicies = {}
    const {taskEmbeddings, stateEmbeddings, actionEmbeddings} = getEmbeddings()
    const actionIndexes = {}
    const {taskIndexes, dataSet} = await preprocessFiles()
    console.log(dataSet)
    const tIndices = Object.keys(taskIndexes)
    const allTasks = {}
    for(const tIndex of tIndices) {
        const desc = createTaskDescriptions(tIndex, taskIndexes, dataSet)
        allTasks[tIndex] = desc.task
    }
    for(const sample of dataSet) {
        const seqLength = sample[1][0].length
        if(!(seqLength in actionDistribution)) {
            actionDistribution[seqLength] = []
        }
        actionDistribution[seqLength].push(sample[0])
    }
    const randomlySelectedTasks = []
    for(let j = 0; j < numRuns; j++) {
        for (const seqSize in actionDistribution) {
            const index = getRandomInt(0, actionDistribution[seqSize].length)
            const t = actionDistribution[seqSize][index]
            if (!(randomlySelectedTasks.includes(t))) {
                randomlySelectedTasks.push(t)
                actionDistribution[seqSize].splice(index, 1)
                if (actionDistribution[seqSize].length <= 0)
                    delete actionDistribution[seqSize]
            } else {
                for (const tsk of actionDistribution[seqSize]) {
                    randomlySelectedTasks.push(tsk)
                    actionDistribution[seqSize].shift()
                    if (actionDistribution[seqSize].length <= 0)
                        delete actionDistribution[seqSize]
                }
            }
        }
    }
    const execList = []
    for(const t of randomlySelectedTasks) {
        const randomTask = t
        const task = allTasks[randomTask]
        execList.push(getPoliciesPerTask(task.id, distLimit, dataSet, stateEmbeddings, actionEmbeddings))
    }
    const results = await Promise.all(execList)
    for(const result of results) {
        const {taskID, stateList, perimeters, steps, cumRewards, wrongDecisions} = result
        console.log(perimeters)
        composedPolicies[taskID] = {
            policies: stateList,
            steps: steps + 1,
            maxDistance: perimeters !== undefined && perimeters.length > 0 ? (perimeters.reduce((a, b) => a + b) / perimeters.length) : 0.25,
            policiesLength: stateList.length - 1
        }
        csv += `${taskID},${cumRewards},`
        csv += `${wrongDecisions},${composedPolicies[taskID].steps},${composedPolicies[taskID].maxDistance},${composedPolicies[taskID].policiesLength},1\n`
    }

    writeFileSync(`./ComposedPolicies.json`, JSON.stringify(composedPolicies, null, 2))
    writeFileSync('./AgentEnsemblesEval.csv', csv)
}

async function getPoliciesPerTask(taskName, distLimit, dataSet, stateEmbeddings, actionEmbeddings) {
    const states = []
    const agents = []
    const taskID = taskName
    const policies = dataSet.filter((a) => a[0] === taskID)
    const originalT = getTaskProfile("../VirtualHomeKG/JSONLD/" + taskID + '.jsonld')
    const requesterAgent = {
        id: "RequestAgent", state: undefined, reward: 0, policies: policies[0][1], actionIndex: 1}
    const startAction = requesterAgent.policies[0][0]
    const sendMsg = JSON.stringify({
        timestamp: new Date(),
        status: 'training',
        action: originalT.actions[startAction],
        step: requesterAgent.actionIndex
    })
    let feedback = JSON.parse(simulateLocal(originalT, false, requesterAgent.id, sendMsg, undefined))
    const stateList = []
    stateList.push({
        state: reasonState(originalT, feedback.lastSendState.unscaledVector)[0],
        features: JSON.parse(JSON.stringify(feedback.lastSendState.unscaledVector)),
        action: Object.keys(feedback.lastAction)[0],
        reward: feedback.reward
    })
    let goalState = undefined
    let initialState = requesterAgent.policies[0][0] + "_Done"
    for (const state in originalT.states) {
        if (originalT.states[state][state].IsGoal) {
            goalState = state
        }
    }
    let currentState = initialState
    let startFlag = true
    let lastState = initialState
    let lastReward = 0.25
    let steps = 0
    let maxDist = distLimit
    let cumRewards = 0
    let wrongDecisions = 0
    const perimeters = []
    while (!feedback.goal) {
        const stateVector = stateEmbeddings[currentState]
        const actionNeighbours = []
        const stateNeighbours = []
        for (const stat in stateEmbeddings) {
            if (stat !== currentState) {
                const stateVector2 = stateEmbeddings[stat]
                const cossim = cosineSimilarity(stateVector, stateVector2)
                const cosDistance = 1.0 - cossim
                stateNeighbours.push([stat, cosDistance])
            }
        }
        stateNeighbours.sort((a, b) => a[1] - b[1])
        const closestStates = stateNeighbours.filter((a) => a[1] <= maxDist) //(stateNeighbours.length >= numEnsembles) ? stateNeighbours.slice(0, numEnsembles) : stateNeighbours
        //TODO: Go through all closestStates and find the nearest action

        for (const act in actionEmbeddings) {
            const actionVector = actionEmbeddings[act]
            const cossim = cosineSimilarity(stateVector, actionVector)
            const cosDistance = 1.0 - cossim
            actionNeighbours.push([act, cosDistance])
        }
        actionNeighbours.sort((a, b) => a[1] - b[1])

        const closestActions = actionNeighbours.filter((a) => a[1] <= maxDist) //(actionNeighbours.length >= numEnsembles) ? actionNeighbours.slice(0, numEnsembles) : actionNeighbours
        if (closestActions.length > 0) {
            let c = 0
            for (const act of closestActions) {
                const agent = {
                    id: 'Agent' + c, policies: act, state: undefined, reward: 0, actionIndex: 1}
                agents.push(agent)
                c += 1
            }
            const threads = []
            for (const a of agents) {
                const action = originalT.actions[a.policies[0]]
                if (action !== undefined) {
                    const sendMsg = JSON.stringify({
                        timestamp: new Date(),
                        status: 'training',
                        action: action,
                        step: a.actionIndex
                    })
                    threads.push(simulateLocal(originalT, false, a.id, sendMsg, feedback))
                    a.actionIndex += 1
                } else {
                    wrongDecisions += 1
                }
            }
            agents.length = 0
            const feedbacks = threads.map(f => JSON.parse(f))
            if(feedbacks.length > 0) {
                feedbacks.sort((a, b) => b.reward - a.reward)
                cumRewards += feedbacks[0].reward
            }
            if (feedbacks.length > 0 && feedbacks[0].reward > lastReward) {
                feedback = feedbacks[0]
                const reasonedStates = reasonState(originalT, feedback.lastSendState.unscaledVector)
                for (const st of reasonedStates) {
                    if (!states.includes(st)) {
                        states.push(st)
                    }
                }
                currentState = states[states.length - 1]
                if (currentState !== lastState) {
                    startFlag = false
                    stateList.push({
                        state: undefined,
                        features: JSON.parse(JSON.stringify(feedback.lastSendState.unscaledVector)),
                        action: Object.keys(feedbacks[0].lastAction)[0],
                        reward: feedbacks[0].reward
                    })
                    lastReward = feedbacks[0].reward
                    perimeters.push(maxDist)
                    maxDist = distLimit
                } else {
                    if (!startFlag) {
                        maxDist += 0.25
                        console.log(maxDist)
                    }
                    if (startFlag) {
                        startFlag = false
                        stateList.push({
                            state: undefined,
                            features: JSON.parse(JSON.stringify(feedback.lastSendState.unscaledVector)),
                            action: Object.keys(feedbacks[0].lastAction)[0],
                            reward: feedbacks[0].reward
                        })
                        lastReward = feedbacks[0].reward
                    }
                }
                lastState = currentState
            } else {
                wrongDecisions += 1
                maxDist += 0.25
            }
        } else {
            maxDist += 0.25

        }
        steps += 1
        actionNeighbours.length = 0
        stateNeighbours.length = 0
    }
    const finalState = {
        state: states[states.length - 1],
        features: feedback.lastSendState.unscaledVector,
        action: "Stop",
        reward: stateList[stateList.length - 1].reward + 0.25
    }
    stateList.push(finalState)
    for (let m = 0; m < states.length - 1; m++) {
        stateList[m + 1].state = states[m]
    }
    return {taskID, stateList, perimeters, steps, cumRewards, wrongDecisions}
}

main(52, 1, 0.25)