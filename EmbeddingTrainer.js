#!/usr/bin/env node

const tf = require('@tensorflow/tfjs')
const use = require('@tensorflow-models/universal-sentence-encoder')
const {readdirSync, readFileSync, renameSync, writeFileSync, createReadStream} = require('fs')
const readline = require('readline')
const stream = require('stream')
const {getRandomInt} = require('./utils')
require('@tensorflow/tfjs-node')

async function trainAll(taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, dataSet) {
    const actionIndexes = {}
    console.log(dataSet)
    const sortedData = dataSet.map((task) => {
        task[1].sort((a, b) => b.length - a.length)
        return task
    })
    const vocab2Task = {}
    const task2action = {}
    let id = -1
    sortedData.forEach((sample) => {
        task2action[sample[0]] = []
        for (const v of sample[1]) {
            for (const elem of v) {
                task2action[sample[0]].push(elem)
                if (!(elem in actionIndexes)) {
                    id += 1
                    actionIndexes[elem] = id
                    vocab2Task[elem] = []
                }
                if (!(vocab2Task[elem].includes(sample[0])))
                    vocab2Task[elem].push(sample[0])
            }
        }
    })
    const vocabSize = Object.keys(actionIndexes).length
    const embeddingDimSize = 50
    const samples = createTrainingData(sortedData, actionIndexes, vocab2Task, task2action, taskIndexes, taskIndexArray, stateIndexes)
    const model = createNetworkMerged(embeddingDimSize, Object.keys(stateIndexes).length,
        Object.keys(actionIndexes).length, Object.keys(taskIndexes).length, "stateEmbedding", "actionEmbedding", "taskEmbedding")
    const range = Array.from(new Array(1000))
    for(const iter of range) {
        const batch = generateBatches/*4Tasks*/(samples, taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, 1024, 1).next()
        await trainEmbeddingsMerged(model, batch.value.input1, batch.value.input2, batch.value.input3, batch.value.output1,
                batch.value.output2, batch.value.output3, batch.value.batches, 15)
    }
    const stateEmbedding = await model.getLayer('stateEmbedding').getWeights()[0].array()
    const actionEmbedding = await model.getLayer('actionEmbedding').getWeights()[0].array()
    const taskEmbedding = await model.getLayer('taskEmbedding').getWeights()[0].array()

    let taskEmbeddingsTSV = ``
    let taskMetaTSV = ``
    let tCounter = 0
    for (const tIndex in taskIndexes) {
        const index = taskIndexes[tIndex]
        taskEmbeddingsTSV += taskEmbedding[index].reduce((a, b) => `${a}\t${b}`)
        taskEmbeddingsTSV += `\n`
        if (tCounter === index)
            taskMetaTSV += `${tIndex}\n`
        tCounter += 1
    }

    let stateEmbeddingsTSV = ``
    let stateMetaTSV = ``
    let sCounter = 0
    for (const sIndex in stateIndexes) {
        const index = stateIndexes[sIndex]
        stateEmbeddingsTSV += stateEmbedding[index].reduce((a, b) => `${a}\t${b}`)
        stateEmbeddingsTSV += `\n`
        if (sCounter === index)
            stateMetaTSV += `${sIndex}\n`
        sCounter += 1
    }

    let actionEmbeddingsTSV = ``
    let actionMetaTSV = ``
    let aCounter = 0
    for (const aIndex in actionIndexes) {
        const index = actionIndexes[aIndex]
        actionEmbeddingsTSV += actionEmbedding[index].reduce((a, b) => `${a}\t${b}`)
        actionEmbeddingsTSV += `\n`
        if (aCounter === index)
            actionMetaTSV += `${aIndex}\n`
        aCounter += 1
    }
    writeFileSync('./taskEmbeddingsMerged.tsv', taskEmbeddingsTSV)
    writeFileSync('./taskMetaTSVMerged.tsv', taskMetaTSV)
    writeFileSync('./actionEmbeddingsMerged.tsv', actionEmbeddingsTSV)
    writeFileSync('./actionMetaTSVMerged.tsv', actionMetaTSV)
    writeFileSync('./stateEmbeddingsMerged.tsv', stateEmbeddingsTSV)
    writeFileSync('./stateMetaTSVMerged.tsv', stateMetaTSV)

    let consolidatedEmbeddings = taskEmbeddingsTSV += actionEmbeddingsTSV += stateEmbeddingsTSV
    let consolidatedLabels = taskMetaTSV += actionMetaTSV += stateMetaTSV
    writeFileSync('./consolidated_embeddings_merged.tsv', consolidatedEmbeddings)
    writeFileSync('./consolidated_labels_merged.tsv', consolidatedLabels)
    return {taskEmbedding, actionEmbedding, stateEmbedding, sortedData, consolidatedEmbeddings, consolidatedLabels}
}

async function trainTwo(taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, dataSet) {
    const actionIndexes = {}
    console.log(dataSet)
    const sortedData = dataSet.map((task) => {
        task[1].sort((a, b) => b.length - a.length)
        return task
    })
    const vocab2Task = {}
    const task2action = {}
    let id = -1
    sortedData.forEach((sample) => {
        task2action[sample[0]] = []
        for (const v of sample[1]) {
            for (const elem of v) {
                task2action[sample[0]].push(elem)
                if (!(elem in actionIndexes)) {
                    id += 1
                    actionIndexes[elem] = id
                    vocab2Task[elem] = []
                }
                if (!(vocab2Task[elem].includes(sample[0])))
                    vocab2Task[elem].push(sample[0])
            }
        }
    })
    const vocabSize = Object.keys(actionIndexes).length
    const embeddingDimSize = 50
    const samples = createTrainingData(sortedData, actionIndexes, vocab2Task, task2action, taskIndexes, taskIndexArray, stateIndexes)
    const model1 = createNetwork(embeddingDimSize, Object.keys(stateIndexes).length, Object.keys(actionIndexes).length, "stateEmbedding","actionEmbedding")
    const model2 = createNetwork(embeddingDimSize, Object.keys(stateIndexes).length, Object.keys(taskIndexes).length, "stateEmbedding", "taskEmbedding")
    const model3 = createNetwork(embeddingDimSize, Object.keys(taskIndexes).length, Object.keys(actionIndexes).length, "taskEmbedding", "actionEmbedding")
    const execList = []
    const range = Array.from(new Array(1000))
    for(const iter of range) {
        const batch = generateBatches/*4Tasks*/(samples, taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, 1024, 1).next()
        execList.push(trainEmbeddings(model1, batch.value.input1, batch.value.input2, batch.value.output1, batch.value.batches, 15))
        execList.push(trainEmbeddings(model2, batch.value.input1, batch.value.input3, batch.value.output2, batch.value.batches, 15))
        execList.push(trainEmbeddings(model3, batch.value.input3, batch.value.input2, batch.value.output3, batch.value.batches, 15))
        await Promise.all(execList)
        execList.length = 0
    }
    const stateEmbedding2Action = await model1.getLayer('stateEmbedding').getWeights()[0].array()
    const actionEmbedding2State = await model1.getLayer('actionEmbedding').getWeights()[0].array()

    const stateEmbedding2Task = await model2.getLayer('stateEmbedding').getWeights()[0].array()
    const taskEmbedding2State = await model2.getLayer('taskEmbedding').getWeights()[0].array()

    const taskEmbedding2Action = await model3.getLayer('taskEmbedding').getWeights()[0].array()
    const actionEmbedding2Task = await model3.getLayer('actionEmbedding').getWeights()[0].array()

    let taskEmbeddings2ActionTSV = ``
    let taskMetaTSV = ``
    let tCounter = 0
    for (const tIndex in taskIndexes) {
        const index = taskIndexes[tIndex]
        taskEmbeddings2ActionTSV += taskEmbedding2Action[index].reduce((a, b) => `${a}\t${b}`)
        taskEmbeddings2ActionTSV += `\n`
        if (tCounter === index)
            taskMetaTSV += `${tIndex}\n`
        tCounter += 1
    }

    let taskEmbeddings2StateTSV = ``
    for (const tIndex in taskIndexes) {
        const index = taskIndexes[tIndex]
        taskEmbeddings2StateTSV += taskEmbedding2State[index].reduce((a, b) => `${a}\t${b}`)
        taskEmbeddings2StateTSV += `\n`
    }

    let stateEmbeddings2ActionTSV = ``
    let stateMetaTSV = ``
    let sCounter = 0
    for (const sIndex in stateIndexes) {
        const index = stateIndexes[sIndex]
        stateEmbeddings2ActionTSV += stateEmbedding2Action[index].reduce((a, b) => `${a}\t${b}`)
        stateEmbeddings2ActionTSV += `\n`
        if (sCounter === index)
            stateMetaTSV += `${sIndex}\n`
        sCounter += 1
    }

    let stateEmbeddings2TaskTSV = ``
    for (const sIndex in stateIndexes) {
        const index = stateIndexes[sIndex]
        stateEmbeddings2TaskTSV += stateEmbedding2Task[index].reduce((a, b) => `${a}\t${b}`)
        stateEmbeddings2TaskTSV += `\n`
    }

    let actionEmbeddings2TaskTSV = ``
    let actionMetaTSV = ``
    let aCounter = 0
    for (const aIndex in actionIndexes) {
        const index = actionIndexes[aIndex]
        actionEmbeddings2TaskTSV += actionEmbedding2Task[index].reduce((a, b) => `${a}\t${b}`)
        actionEmbeddings2TaskTSV += `\n`
        if (aCounter === index)
            actionMetaTSV += `${aIndex}\n`
        aCounter += 1
    }

    let actionEmbeddings2StateTSV = ``
    for (const aIndex in actionIndexes) {
        const index = actionIndexes[aIndex]
        actionEmbeddings2StateTSV += actionEmbedding2State[index].reduce((a, b) => `${a}\t${b}`)
        actionEmbeddings2StateTSV += `\n`
    }
    writeFileSync('./taskEmbeddings2Action.tsv', taskEmbeddings2ActionTSV)
    writeFileSync('./taskEmbeddings2State.tsv', taskEmbeddings2StateTSV)
    writeFileSync('./taskMetaTSV.tsv', taskMetaTSV)

    writeFileSync('./actionEmbeddings2State.tsv', actionEmbeddings2StateTSV)
    writeFileSync('./actionEmbeddings2Task.tsv', actionEmbeddings2TaskTSV)
    writeFileSync('./actionMetaTSV.tsv', actionMetaTSV)

    writeFileSync('./stateEmbeddings2Action.tsv', stateEmbeddings2ActionTSV)
    writeFileSync('./stateEmbeddings2Task.tsv', stateEmbeddings2TaskTSV)
    writeFileSync('./stateMetaTSV.tsv', stateMetaTSV)

    return {taskEmbedding2Action, taskEmbedding2State, actionEmbedding2Task, actionEmbedding2State,
        stateEmbedding2Action, stateEmbedding2Task, sortedData}
}

async function main() {
    //await renameFileNames()
    const {taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, dataSet} = await preprocessFiles()
    //const result1 = await trainTwo(taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, dataSet)
    const result2 = await trainAll(taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, dataSet)

    //const {taskEmbedding, actionEmbedding, stateEmbedding, sortedData, consolidatedEmbeddings, consolidatedLabels} =
    //const {taskEmbedding, actionEmbedding, stateEmbedding, sortedData, consolidatedEmbeddings, consolidatedLabels} = await
    /*const taskNeighbours = {}
    for (const key1 in taskIndexes) {
        if (!(key1 in taskNeighbours)) {
            taskNeighbours[key1] = []
        }
        for (const key2 in taskIndexes) {
            if (key1 !== key2) {
                const e1Index = taskIndexes[key1]
                const e2Index = taskIndexes[key2]
                const v1 = taskEmbedding[e1Index]
                const v2 = taskEmbedding[e2Index]
                const cossim = cosineSimilarity(v1, v2)
                const cosDistance = 1.0 - cossim
                taskNeighbours[key1].push([key2, cosDistance])
            }
        }
    }
    for (const key in taskNeighbours) {
        taskNeighbours[key].sort((a, b) => a[1] - b[1])
    }
    console.log(taskNeighbours)
    const tasks = Object.keys(taskNeighbours)
    for (let i = 0; i < 5; i++) {
        const index = getRandomInt(0, tasks.length)
        const tName = tasks[index]
        const neighbour = taskNeighbours[tName][0][0]
        const task1 = sortedData.filter((d) => d[0] == tName)
        const task2 = sortedData.filter(d => d[0] == neighbour)
        const seq1 = task1[0][1]
        console.log(tName)
        const out1 = seq1[0].reduce((a, b) => a + ", " + b)
        const seq2 = task2[0][1]
        console.log(neighbour)
        console.log("Cosine Distance: "+taskNeighbours[tName][0][1])
        const out2 = seq2[0].reduce((a, b) => a + ", " + b)
        const matches = seq1[0].filter(s1 => seq2[0].includes(s1))
        console.log(`Matches: ${matches.length}`)
        console.log(`Action Sequence length 1: ${seq1[0].length}`)
        console.log(`Action Sequence length 2: ${seq2[0].length}`)
        console.log(`Match-Rate1: ${matches.length / seq1[0].length}`)
        console.log(`Match-Rate2: ${matches.length / seq2[0].length}`)
        const consolidatedActions = []
        seq1[0].forEach((el) => {
            if (!(consolidatedActions.includes(el))) {
                consolidatedActions.push(el)
            }
        })
        seq2[0].forEach((el) => {
            if (!(consolidatedActions.includes(el))) {
                consolidatedActions.push(el)
            }
        })
        console.log(`Jaccard-Similarity of Actions: ${matches.length / (consolidatedActions.length)}`)
        console.log("-----")
        //Generate of the two nearest tasks a task spec
        const task_consolidated = {
            id: `${tName}_${neighbour}_consolidated`,
            sequential: true,
            communicationType: 'Asynchronized',
            numberOfActors: 1,
            features: [],
            states: [],
            actions: [],
            topics: [],
            effects: {}
        }
        const t1 = {
            id: tName,
            sequential: true,
            communicationType: 'Asynchronized',
            numberOfActors: 1,
            features: [],
            states: [],
            actions: [],
            topics: [],
            effects: {}
        }
        const t2 = {
            id: neighbour,
            sequential: true,
            communicationType: 'Asynchronized',
            numberOfActors: 1,
            features: [],
            states: [],
            actions: [],
            topics: [],
            effects: {}
        }
        const topicId = `TaskTopic`
        const topic = {id: topicId, features: []}
        let featTracker = []
        const initState = {
            id: 'InitialState',
            expression: '',
            actions: [],
            features: [],
            goal: false,
            initial: true,
            reward: 0
        }

        consolidatedActions.forEach((act) => {
            const stateId = `${act}_Done`
            const featId = `Is${act}`
            featTracker.push(featId)
            const actionId = `${act}`
            const effectId = `Set${act}`
            let g = false
            let rew = 0
            let exp = ``
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
            if (actionId === seq1[0][seq1[0].length-1] || actionId === seq2[0][seq2[0].length-1]) {
                g = true
                rew = 1
                let initExp = `(`
                featTracker.forEach((f) => {
                    initExp += `${f} == 0 AND `
                })
                initExp = initExp.substring(0, initExp.lastIndexOf("AND")).trim() + ") OR "
                initState.expression += initExp
                featTracker.length = 0
            }
            state.goal = g
            state.initial = false
            state.reward = rew
            initState.actions.push(actionId)
            initState.features.push(featId)
            const action = {id: `${actionId}`, effects: [effectId], negation: false, text: ""}
            const feat = {id: featId, rangeStart: 0, rangeEnd: 1, type: 'NOMINAL', unit: ""}
            const effect = {id: effectId, impactType: "ON", features: [featId]}
            topic.features.push(feat.id)
            task_consolidated.states.push(state)
            task_consolidated.actions.push(action)
            task_consolidated.features.push(feat)
            task_consolidated.effects[effectId] = effect
        })
        initState.expression = initState.expression.substring(0, initState.expression.lastIndexOf("OR")).trim()
        task_consolidated.topics.push(topic)
        task_consolidated.states.push(initState)

        featTracker.length = 0
        let count = 0
        const initState1 = {
            id: 'InitialState',
            expression: '',
            actions: [],
            features: [],
            goal: false,
            initial: true,
            reward: 0
        }
        const topic1ID = `TaskTopic`
        const topic1 = {id: topic1ID, features: []}
        seq1[0].forEach((act) => {
            const stateId = `${act}_Done`
            const featId = `Is${act}`
            featTracker.push(featId)
            const actionId = `${act}`
            const effectId = `Set${act}`
            let g = false
            let rew = 0
            let exp = ``
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
            if (count == seq1[0].length - 1) {
                g = true
                let initExp = ``
                featTracker.forEach((f) => {
                    initExp += `${f} == 0 AND `
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
            const action = {id: `${actionId}`, effects: [effectId], negation: false, text: ""}
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

        featTracker.length = 0
        count = 0
        const topic2Id = `TaskTopic`
        const topic2 = {id: topic2Id, features: []}
        const initState2 = {
            id: 'InitialState',
            expression: '',
            actions: [],
            features: [],
            goal: false,
            initial: true,
            reward: 0
        }
        seq2[0].forEach((act) => {
            const stateId = `${act}_Done`
            const featId = `Is${act}`
            featTracker.push(featId)
            const actionId = `${act}`
            const effectId = `Set${act}`
            let g = false
            let rew = 0
            let exp = ``
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
            if (count == seq2[0].length - 1) {
                g = true
                rew = 1
                let initExp = ``
                featTracker.forEach((f) => {
                    initExp += `${f} == 0 AND `
                })
                initExp = initExp.substring(0, initExp.lastIndexOf("AND")).trim()
                initState2.expression += initExp
                featTracker.length = 0
            }
            state.goal = g
            state.initial = false
            state.reward = rew
            initState2.actions.push(actionId)
            initState2.features.push(featId)
            const action = {id: `${actionId}`, effects: [effectId], negation: false, text: ""}
            const feat = {id: featId, rangeStart: 0, rangeEnd: 1, type: 'NOMINAL', unit: ""}
            const effect = {id: effectId, impactType: "ON", features: [featId]}
            topic2.features.push(feat.id)
            t2.states.push(state)
            t2.actions.push(action)
            t2.features.push(feat)
            t2.effects[effectId] = effect
            count += 1
        })
        t2.topics.push(topic2)
        t2.states.push(initState2)
        const semTask = semantifyAndStoreTask(task_consolidated)
        const taskJson = JSON.stringify(semTask, null, 2)
        //writeFileSync(`${tName}_${neighbour}_consolidated.jsonld`, taskJson)

        const semT1 = semantifyAndStoreTask(t1)
        const t1Json = JSON.stringify(semT1.jsonld, null, 2)
        //writeFileSync(`${tName}.jsonld`, t1Json)
        writeFileSync(`${tName}.ttl`, semT1.turtle)

        const semT2 = semantifyAndStoreTask(t2)
        const t2Json = JSON.stringify(semT2.jsonld, null, 2)
        //writeFileSync(`${neighbour}.jsonld`, t2Json)
        writeFileSync(`${neighbour}.ttl`, semT2.turtle)
    }*/
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
        let act = {
            '@id': entity+action.id,
            '@type': [concept+'Action'],
            [prop+'HasText']: [{'@value': action.text}],
           // [prop+'IsNegation']: [{'@type': xmlSchema+'boolean', '@value': action.negation}],
            [prop+'HasEffect']: []
        }
        turtle += `entity:${action.id} a concept:Action;\n`
        turtle += `\tproperty:HasText "${action.text}"^^xsd:string;\n`
        turtle += `\tproperty:IsNegation "${action.negation}"^^xsd:boolean;\n`
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

async function trainEmbeddingsMerged(model, x1, x2, x3, y1, y2, y3, batchSize, epochs) {
    const result = await model.fit({['input1']: x1, ['input2']: x2, ['input3']: x3}, [y1, y2, y3], {
        epochs: epochs,
        shuffle: false,
        batchSize: batchSize
    })
}

async function trainEmbeddings(model, x1, x2, y, batchSize, epochs) {
    const result = await model.fit({['input1']: x1, ['input2']: x2}, y, {
        epochs: epochs,
        shuffle: false,
        batchSize: batchSize
    })
}

function* generateBatches(samples, taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, numPositiveExamples, ratioNegativeExamples) {
    const samplesCopy = samples.slice()
    tf.util.shuffle(samplesCopy)
    const batchSize = numPositiveExamples * (1 + ratioNegativeExamples)

    const positiveExamplesTaskAction = samplesCopy.filter(sCopy => sCopy.isTask === 1 && sCopy.isAction === 1)
    const positiveExamplesTaskState = samplesCopy.filter(sCopy => sCopy.isTask === 1 && sCopy.isState === 1)
    const positiveExamplesStateAction = samplesCopy.filter(sCopy => sCopy.isState === 1 && sCopy.isAction === 1)
    const negativeExamplesTaskAction = samplesCopy.filter(sCopy => sCopy.isTask === 1 && sCopy.isAction === 0 || (sCopy.isTask === 0 && sCopy.isAction === 1))
    const negativeExamplesTaskState = samplesCopy.filter(sCopy => sCopy.isTask === 1 && sCopy.isState === 0 || (sCopy.isTask === 0 && sCopy.isState === 1))
    const negativeExamplesStateAction = samplesCopy.filter(sCopy => sCopy.isState === 1 && sCopy.isAction === 0 || sCopy.isState === 0 && sCopy.isAction === 1)
    while(true) {
        const batch1 = []
        const batch2 = []
        const batch3 = []
        const labels1 = []
        const labels2 = []
        const labels3 = []
        let batches = 0
        while (batches < batchSize) {
            for (let i = 0; i < numPositiveExamples; i++) {
                const randIndexPosTaskAction = getRandomInt(0, positiveExamplesTaskAction.length)
                const randIndexPosTaskState = getRandomInt(0, positiveExamplesTaskState.length)
                const randIndexPosStateAction = getRandomInt(0, positiveExamplesStateAction.length)
                //Positive examples State and Action
                batch1.push(positiveExamplesStateAction[randIndexPosStateAction].stateIndex)
                batch2.push(positiveExamplesStateAction[randIndexPosStateAction].actionIndex)
                batch3.push(positiveExamplesStateAction[randIndexPosStateAction].taskIndex)
                labels1.push(1)
                const l2 = positiveExamplesStateAction[randIndexPosStateAction].isState === 1 && positiveExamplesStateAction[randIndexPosStateAction].isTask === 1 ? 1 : 0
                labels2.push(l2)
                const l3 = positiveExamplesStateAction[randIndexPosStateAction].isAction === 1 && positiveExamplesStateAction[randIndexPosStateAction].isTask === 1 ? 1 : 0
                labels3.push(l3)

                //Positive Examples Task and State
                batch1.push(positiveExamplesTaskState[randIndexPosTaskState].stateIndex)
                batch2.push(positiveExamplesTaskState[randIndexPosTaskState].actionIndex)
                batch3.push(positiveExamplesTaskState[randIndexPosTaskState].taskIndex)
                const l1 = positiveExamplesTaskState[randIndexPosTaskState].isState === 1 && positiveExamplesTaskState[randIndexPosTaskState].isAction === 1 ? 1 : 0
                labels1.push(l1) //state and action
                labels2.push(1) //task and state
                const l31 = positiveExamplesTaskState[randIndexPosTaskState].isTask === 1 && positiveExamplesTaskState[randIndexPosTaskState].isAction === 1 ? 1 : 0
                labels3.push(l31) //task and action

                //Positive Examples Task and Action
                batch1.push(positiveExamplesTaskAction[randIndexPosTaskAction].stateIndex)
                batch2.push(positiveExamplesTaskAction[randIndexPosTaskAction].actionIndex)
                batch3.push(positiveExamplesTaskAction[randIndexPosTaskAction].taskIndex)
                const l12 = positiveExamplesTaskAction[randIndexPosTaskAction].isState === 1 && positiveExamplesTaskAction[randIndexPosTaskAction].isAction === 1 ? 1 : 0
                labels1.push(l12)
                const l22 = positiveExamplesTaskAction[randIndexPosTaskAction].isState === 1 && positiveExamplesTaskAction[randIndexPosTaskAction].isTask === 1 ? 1 : 0
                labels2.push(l22)
                labels3.push(1) //task and action
                batches += 3
            }
            const randIndexNegTaskAction = getRandomInt(0, negativeExamplesTaskAction.length)
            const randIndexNegTaskState = getRandomInt(0, negativeExamplesTaskState.length)
            const randIndexNegStateAction = getRandomInt(0, negativeExamplesStateAction.length)

            //Negative examples
            const l2N = negativeExamplesStateAction[randIndexNegStateAction].isState === 1 && negativeExamplesStateAction[randIndexNegStateAction].isTask === 1 ? 1 : 0
            const l3N = negativeExamplesStateAction[randIndexNegStateAction].isAction === 1 && negativeExamplesStateAction[randIndexNegStateAction].isTask === 1 ? 1 : 0
            batch1.push(negativeExamplesStateAction[randIndexNegStateAction].stateIndex)
            batch2.push(negativeExamplesStateAction[randIndexNegStateAction].actionIndex)
            batch3.push(negativeExamplesStateAction[randIndexNegStateAction].taskIndex)
            labels1.push(0)
            labels2.push(l2N)
            labels3.push(l3N)

            //Negative Examples Task and State
            batch1.push(negativeExamplesTaskState[randIndexNegTaskState].stateIndex)
            batch2.push(negativeExamplesTaskState[randIndexNegTaskState].actionIndex)
            batch3.push(negativeExamplesTaskState[randIndexNegTaskState].taskIndex)
            const l1N = negativeExamplesTaskState[randIndexNegTaskState].isState === 1 && negativeExamplesTaskState[randIndexNegTaskState].isAction === 1 ? 1 : 0
            labels1.push(l1N) //state and action
            labels2.push(0) //task and state
            const l31N = negativeExamplesTaskState[randIndexNegTaskState].isTask === 1 && negativeExamplesTaskState[randIndexNegTaskState].isAction === 1 ? 1 : 0
            labels3.push(l31N) //task and action

            //Positive Examples Task and Action
            batch1.push(negativeExamplesTaskAction[randIndexNegTaskAction].stateIndex)
            batch2.push(negativeExamplesTaskAction[randIndexNegTaskAction].actionIndex)
            batch3.push(negativeExamplesTaskAction[randIndexNegTaskAction].taskIndex)
            const l12N = negativeExamplesTaskAction[randIndexNegTaskAction].isState === 1 && negativeExamplesTaskAction[randIndexNegTaskAction].isAction === 1 ? 1 : 0
            labels1.push(l12N)
            const l22N = negativeExamplesTaskAction[randIndexNegTaskAction].isState === 1 && negativeExamplesTaskAction[randIndexNegTaskAction].isTask === 1 ? 1 : 0
            labels2.push(l22N)
            labels3.push(0) //task and action
            batches += 3
        }
        const input1 = tf.tensor1d(batch1) //state
        const input2 = tf.tensor1d(batch2) //action
        const input3 = tf.tensor1d(batch3) //task
        const output1 = tf.tensor1d(labels1) //state and action
        const output2 = tf.tensor1d(labels2) //state and task
        const output3 = tf.tensor1d(labels3) // task and action
        yield {input1, input2, input3, output1, output2, output3, batches}
    }
}

function* generateBatches4Tasks(samples, taskName, taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, numPositiveExamples, ratioNegativeExamples) {
    const batch1 = []
    const batch2 = []
    const batch3 = []
    const labels1 = []
    const labels2 = []
    const labels3 = []
    let batches = 0
    const samplesCopy = samples.slice()
    tf.util.shuffle(samplesCopy)
    const batchSize = numPositiveExamples * (1 + ratioNegativeExamples)
   // for (const t in taskIndexes) {
       // const tIndex = getRandomInt(0, taskIndexArray.length)
       // const t = taskIndexArray[tIndex]
        const positiveExamples = samplesCopy.filter(sCopy => sCopy.task === taskName && sCopy.isTask === 1 && sCopy.isAction === 1)
        const negativeExamples = samplesCopy.filter(sCopy => sCopy.task === taskName && sCopy.isTask === 1 && sCopy.isAction === 0)
        const size = positiveExamples.length < (batchSize / (1 + ratioNegativeExamples)) ? positiveExamples.length : (batchSize / (1 + ratioNegativeExamples))
        for (let i = 0; i < size; i++) {
            const randIndexPos = getRandomInt(0, positiveExamples.length)
            const randIndexNeg = getRandomInt(0, negativeExamples.length)
            //Positive examples
            batch1.push(positiveExamples[randIndexPos].stateIndex)
            batch2.push(positiveExamples[randIndexPos].actionIndex)
            batch3.push(positiveExamples[randIndexPos].taskIndex)
            if(positiveExamples[randIndexPos].isAction === 1 && positiveExamples[randIndexPos].isState === 1) //action and state
                labels1.push(1)
            else
                labels1.push(0)
            labels2.push(positiveExamples[i].isState) //task and state
            labels3.push(1) //task and action
            batches += 1

            //Negative examples
            if(negativeExamples.length >= size) {
                batch1.push(negativeExamples[randIndexNeg].stateIndex)
                batch2.push(negativeExamples[randIndexNeg].actionIndex)
                batch3.push(negativeExamples[randIndexNeg].taskIndex)
                if(negativeExamples[randIndexNeg].isAction === 1 && negativeExamples[randIndexNeg].isState === 1) //action and state
                    labels1.push(1)
                else
                    labels1.push(0)
                labels2.push(negativeExamples[randIndexNeg].isState)
                labels3.push(0)
                batches += 1
            }
        }
   // }

    const input1 = tf.tensor1d(batch1) //state
    const input2 = tf.tensor1d(batch2) //action
    const input3 = tf.tensor1d(batch3) //task
    const output1 = tf.tensor1d(labels1) //state and action
    const output2 = tf.tensor1d(labels2) //state and task
    const output3 = tf.tensor1d(labels3) // task and action
    yield {input1, input2, input3, output1, output2, output3, batches}

}

function* generateBatches4States(samples, stateName, taskIndexes, taskIndexArray, stateIndexes, stateIndexArray, numPositiveExamples, ratioNegativeExamples) {
    const batch1 = []
    const batch2 = []
    const batch3 = []
    const labels1 = []
    const labels2 = []
    const labels3 = []
    let batches = 0
    const samplesCopy = samples.slice()
    tf.util.shuffle(samplesCopy)
    const batchSize = numPositiveExamples * (1 + ratioNegativeExamples)
    //const sIndex = getRandomInt(0, stateIndexArray.length)
    //const stateName = stateIndexArray[sIndex]
    const posExamples = samplesCopy.filter(c => c.state === stateName && c.isState === 1 && c.isAction === 1)
    const negExamples = samplesCopy.filter(c => c.state === stateName && c.isState === 1 && c.isAction === 0)
    const len = (posExamples.length < (batchSize / (1 + ratioNegativeExamples))) ? posExamples.length : (batchSize / (1 + ratioNegativeExamples))
    for (let i = 0; i < len; i++) {
        const randIndexPos = getRandomInt(0, posExamples.length)
        const randIndexNeg = getRandomInt(0, negExamples.length)
        //Positive examples
        batch1.push(posExamples[randIndexPos].stateIndex) //state
        batch2.push(posExamples[randIndexPos].actionIndex) //action
        batch3.push(posExamples[randIndexPos].taskIndex) //task
        labels1.push(posExamples[randIndexPos].isAction) //action
        labels2.push(posExamples[randIndexPos].isTask) //task
        if(posExamples[randIndexPos].isTask === 1 && posExamples[randIndexPos].isAction === 1)
            labels3.push(1)
        else
            labels3.push(0)
        batches += 1

        //Negative examples
        batch1.push(negExamples[randIndexNeg].stateIndex)
        batch2.push(negExamples[randIndexNeg].actionIndex)
        batch3.push(negExamples[randIndexNeg].taskIndex)
        labels1.push(0)
        labels2.push(negExamples[randIndexNeg].isTask)
        if(negExamples[randIndexNeg].isTask === 1 && negExamples[randIndexNeg].isAction === 1)
            labels3.push(1)
        else
            labels3.push(0)
        batches += 1

    }
    const input1 = tf.tensor1d(batch1) //state
    const input2 = tf.tensor1d(batch2) //action
    const input3 = tf.tensor1d(batch3) //task
    const output1 = tf.tensor1d(labels1) //state and action
    const output2 = tf.tensor1d(labels2) //state and task
    const output3 = tf.tensor1d(labels3) // task and action
    yield {input1, input2, input3, output1, output2, output3, batches}
}

function createNetwork(embeddingDimSize, indexesLength1, indexesLength2, embedding1Name, embedding2Name) {
    const input1 = tf.input({name: 'input1', shape: [1]})
    const embedding1 = tf.layers.embedding({name: embedding1Name, inputDim: indexesLength1, outputDim: embeddingDimSize}).apply(input1)

    const input2 = tf.input({name: 'input2', shape: [1]})
    const embedding2 = tf.layers.embedding({name: embedding2Name, inputDim: indexesLength2, outputDim: embeddingDimSize}).apply(input2)

    const dotLayer = tf.layers.dot({name: 'dot_product', normalize: true, axes: 2}).apply([embedding1, embedding2])

    const reshapeLayer = tf.layers.reshape({targetShape: [1]}).apply(dotLayer)

    const denseLayer = tf.layers.dense({name: "output", activation: "sigmoid", units: 1, inputShape: [1]}).apply(reshapeLayer)

    const model = tf.model({name: "myEmbeddings", inputs: [input1, input2], outputs: denseLayer})
    model.compile({optimizer: tf.train.adam(0.001), loss: 'binaryCrossentropy', metrics: ['accuracy']})
    return model
}

function createNetworkMerged(embeddingDimSize, indexesLength1, indexesLength2, indexesLength3, embedding1Name, embedding2Name, embedding3Name) {
    const input1 = tf.input({name: 'input1', shape: [1]})
    const embedding1 = tf.layers.embedding({name: embedding1Name, inputDim: indexesLength1, outputDim: embeddingDimSize}).apply(input1)

    const input2 = tf.input({name: 'input2', shape: [1]})
    const embedding2 = tf.layers.embedding({name: embedding2Name, inputDim: indexesLength2, outputDim: embeddingDimSize}).apply(input2)

    const input3 = tf.input({name: 'input3', shape: [1]})
    const embedding3 = tf.layers.embedding({name: embedding3Name, inputDim: indexesLength3, outputDim: embeddingDimSize}).apply(input3)

    const dotLayer1 = tf.layers.dot({name: 'dot_product1', normalize: true, axes: 2}).apply([embedding1, embedding2])
    const dotLayer2 = tf.layers.dot({name: 'dot_product2', normalize: true, axes: 2}).apply([embedding1, embedding3])
    const dotLayer3 = tf.layers.dot({name: 'dot_product3', normalize: true, axes: 2}).apply([embedding2, embedding3])

    const reshapeLayer1 = tf.layers.reshape({targetShape: [1]}).apply(dotLayer1)
    const reshapeLayer2 = tf.layers.reshape({targetShape: [1]}).apply(dotLayer2)
    const reshapeLayer3 = tf.layers.reshape({targetShape: [1]}).apply(dotLayer3)

    const denseLayer1 = tf.layers.dense({name: "output1", activation: "sigmoid", units: 1, inputShape: [1]}).apply(reshapeLayer1)
    const denseLayer2 = tf.layers.dense({name: "output2", activation: "sigmoid", units: 1, inputShape: [1]}).apply(reshapeLayer2)
    const denseLayer3 = tf.layers.dense({name: "output3", activation: "sigmoid", units: 1, inputShape: [1]}).apply(reshapeLayer3)

    const model = tf.model({name: "myEmbeddings", inputs: [input1, input2, input3], outputs: [denseLayer1, denseLayer2, denseLayer3]})
    model.compile({optimizer: tf.train.adam(0.001), loss: 'binaryCrossentropy', metrics: ['accuracy']})
    return model
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

async function renameFileNames() {
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
            if (lineNumber == 0) {
                task = line
                if(!(task in samples)) {
                    samples[task] = 1
                } else {
                    samples[task] += 1
                }
                break
            }
        }
        const t = task.replace(/ /g, "_")
        renameSync(`${dir}/${file}`, `${dir}/${t}_${samples[task]}.txt`)
    }
}
async function generateTasks() {
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
                const action = `${line.substring(line.indexOf('[')+1, line.indexOf(']'))} ${line.substring(line.indexOf('<')+1, line.indexOf('>'))}`
                samples[task][taskCounter[task]].push(action)
            }
            lineNumber += 1
        }
    }
}

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

function createTrainingData(data, actionIndexes, actions2tasks, task2action, taskIndexes, taskIndexArray, stateIndexes) {
    const samples = []
    for(const task in taskIndexes) {
        for(const action in actions2tasks) {
                if (actions2tasks[action].includes(task)) {
                    samples.push({task: task, action: action, state: action+"_Done", taskIndex: taskIndexes[task],
                        actionIndex: actionIndexes[action], stateIndex: stateIndexes[action+"_Done"], isTask: 1, isAction: 1, isState: 1})
                } else {
                    samples.push({task: task, action: action, state: action+"_Done", taskIndex: taskIndexes[task],
                        actionIndex: actionIndexes[action],stateIndex: stateIndexes[action+"_Done"], isTask: 1, isAction: 0, isState: 0})
                }
        }
    }
    for(const state in stateIndexes) {
        for(const action in actions2tasks) {
            const sub = state.substring(0, state.lastIndexOf('_'))
            const tIndex = getRandomInt(0, taskIndexArray.length)
            const task = taskIndexArray[tIndex]
            if (sub === action) {
                if(task2action[task].includes(sub))
                    samples.push({task: task, action: action, state: state, taskIndex: taskIndexes[task],
                        actionIndex: actionIndexes[action], stateIndex: stateIndexes[state], isTask: 1, isAction: 1, isState: 1})
                else
                    samples.push({task: task, action: action, state: state, taskIndex: taskIndexes[task],
                        actionIndex: actionIndexes[action], stateIndex: stateIndexes[state], isTask: 0, isAction: 1, isState: 1})
            } else {
                if(task2action[task].includes(sub))
                    samples.push({task: task, action: action, state: state, taskIndex: taskIndexes[task],
                        actionIndex: actionIndexes[action], stateIndex: stateIndexes[state], isTask: 1, isAction: 0, isState: 1})
                else
                    samples.push({task: task, action: action, state: state, taskIndex: taskIndexes[task],
                        actionIndex: actionIndexes[action], stateIndex: stateIndexes[state], isTask: 0, isAction: 0, isState: 1})
            }
        }
    }
    return samples
}

async function trainEmbedding(samples) {
    const model = await use.load()
    /*const embeddings = {}
    for(const s of samples){
        const embed = await model.embed(s)
        embeddings[s] = embed
    }*/
    const embeddings = await model.embed(samples)
    const similarity = await computeSimilarity(0, 1, embeddings)
    return similarity
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

function euklideanDiatance(v1, v2) {
    let distance = 0
    if(v1.length === v2.length) {
        for(let i = 0; i < v1.length; i++) {
            distance += Math.pow(v1[i] - v2[i], 2.0)
        }
        distance = Math.sqrt(distance)
        return distance
    }
    return undefined
}

async function computeSimilarity(actionAIndex, actionBIndex, embeddings) {
    const actionAEmbeddings = embeddings.slice([actionAIndex, 0], [1])
    const actionBEmbeddings = embeddings.slice([actionBIndex, 0], [1])
    const actionATranspose = false
    const actionBTranspose = true
    const scoreData = await actionAEmbeddings.matMul(actionBEmbeddings, actionATranspose, actionBTranspose).data()
    return scoreData[0]
}

main()