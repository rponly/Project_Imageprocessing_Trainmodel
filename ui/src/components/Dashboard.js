import React, { Component } from 'react'
import { Container, Row, Col, Card, Button, Modal, FormControl, Spinner } from 'react-bootstrap'
import '../assets/main.css'
import axios from 'axios'

export class Dashboard extends Component {
    constructor(props) {
        super(props)
        this.state = {
            models: [
                {
                    name: "undefined"
                }
            ],
            isModelModalOpen: false,
            isTestModalOpen: false,
            dataToPassToModal: {
                mode: null,
                name: null
            },
            loading: true,
            name: ""
        }
    }

    componentDidMount() {
        this.getModel()
    }

    getModel = () => {
        this.getModelAPI().then(res => {

            console.log(res);
            let models = []

            res.models.forEach(model => {
                models.push({ name: model })
            })

            this.setState({
                models: models
            }, () => this.setState({loading: false}))
        })
    }

    getModelAPI = () => {
        return axios.get("http://127.0.0.1:5000/get").then(res => {
            return res.data
        })
    }

    _handleAddNewModel = () => {
        const formData = new FormData();
        formData.append("model", this.state.name.toLowerCase());
        axios({
            method: "POST",
            url: "http://127.0.0.1:5000/add",
            data: formData,
            headers: {
              "Content-Type": "multipart/form-data"
            }
        }).then(res => {
            console.log(res.data);
            alert(res.data.msg)
            this.getModel()
        })
    }

    toggleModelModal = (data) => {
        if(data)
            this.setState({isModelModalOpen: !this.state.isModelModalOpen, dataToPassToModal: data})
        else
        this.setState({isModelModalOpen: !this.state.isModelModalOpen})
    }

    toggleTestModal = (data) => {
        if(data)
            this.setState({isTestModalOpen: !this.state.isTestModalOpen, dataToPassToModal: data})
        else
        this.setState({isTestModalOpen: !this.state.isTestModalOpen})
    }

    render() {
        if(this.state.loading)
            return (<></>)
        return (
            <Container fluid>
                <ModelModal isOpen={this.state.isModelModalOpen} toggle={this.toggleModelModal} data={this.state.dataToPassToModal}/>
                <TestModal isOpen={this.state.isTestModalOpen} toggle={this.toggleTestModal} data={this.state.dataToPassToModal}/>

                <Row>
                    <Col className="text-center m-1" >
                        <h5>Select Model</h5>
                    </Col>
                </Row>
                <Row>
                    <Col className="justify-content-center d-flex flex-row flex-wrap">
                        { /** list models */}
                        {
                            this.state.models.map((model, index) => {
                                return (
                                    <Card className="m-1 text-center" style={{ width: "250px" }} key={index}>
                                        <Card.Body>
                                            <h6>{model.name.toUpperCase()}</h6>
                                        </Card.Body>
                                        <Card.Footer>
                                            <Button className="mx-1" variant="outline-primary" onClick={() => this.toggleModelModal({mode: 'train', name: model.name})}>Train</Button>
                                            <Button className="mx-1" variant="outline-success" onClick={() => this.toggleTestModal({mode: 'test', name: model.name})}>Test</Button>
                                        </Card.Footer>
                                    </Card>
                                )
                            })
                        }
                        { /** Add new model */}
                            <Card className="m-1 text-center add-new" style={{ width: "250px" }}>
                                <Card.Header><FormControl placeholder="model_name" value={this.state.name} onChange={(e) => this.setState({name: e.target.value})}/></Card.Header>
                                <Card.Body onClick={this._handleAddNewModel}>
                                    <h1>+</h1>
                                </Card.Body>
                            </Card>
                    </Col>
                </Row>
            </Container>
        )
    }
}

class ModelModal extends Component {
    constructor(props) {
        super(props)
        this.state = {
            files: [],
            imagePreviewUrl: [],
            loadingOverlay: false
        }
    }

    _handleSubmit(e) {
        e.preventDefault();

        let genFiles = this.state.genFiles
        let fakeFiles = this.state.fakeFiles

        const formData = new FormData();

        Array.from(genFiles).forEach(file=>{
            formData.append("genFiles", file);
        });

        Array.from(fakeFiles).forEach(file=>{
            formData.append("fakeFiles", file);
        });

        formData.append("model", this.props.data.name);

        this.setState({loadingOverlay: true})

        axios({
            method: "POST",
            url: "http://127.0.0.1:5000/train",
            data: formData,
            headers: {
              "Content-Type": "multipart/form-data"
            }
        }).then(res => {
            console.log(res);
            this.setState({loadingOverlay: false}, () => {
                alert(res.data.msg)
                this._handleClose()
            }
            )
        })
    }

    _handleImageChangeGen(e) {
        e.preventDefault();
        let files = e.target.files;
        console.log(files);
        this.setState({
            genFiles: files
        })
    }

    _handleImageChangeFake(e) {
        e.preventDefault();
        let files = e.target.files;
        console.log(files);
        this.setState({
            fakeFiles: files
        })
    }

    _handleClose = () => {
        this.setState({
            files: [],
            imagePreviewUrl: []
        }, () => this.props.toggle())
    }

    render() {
        return (
            <Modal show={this.props.isOpen} onHide={this._handleClose}>
                {
                    this.state.loadingOverlay ? <LoadingOverlay /> : <></>
                }
                <Modal.Header closeButton>
                <Modal.Title>{ this.props.data.mode != undefined ? this.props.data.mode.toUpperCase() : "" } : { this.props.data.name != undefined ? this.props.data.name.toUpperCase() : "" }</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <form onSubmit={(e)=>this._handleSubmit(e)}>
                        <h5>Gen Signature</h5>
                        <input className="" 
                            type="file" 
                            onChange={(e)=>this._handleImageChangeGen(e)} multiple="multiple"/>
                        <hr />
                        <h5>Fake Signature</h5>
                        <input className="" 
                            type="file" 
                            onChange={(e)=>this._handleImageChangeFake(e)} multiple="multiple"/>
                        <hr />
                        <Button className="float-right" variant="outline-primary" type="submit">
                            { this.props.data.mode != undefined ? this.props.data.mode.toUpperCase() : "undefined" }
                        </Button>
                    </form>
                    
                </Modal.Body>
            </Modal>
        )
    }
}

class TestModal extends Component {
    constructor(props) {
        super(props)
        this.state = {
            files: [],
            imagePreviewUrl: [],
            loadingOverlay: false
        }
    }
    _handleSubmit(e) {
        e.preventDefault();

        let files = this.state.files

        const formData = new FormData();

        Array.from(files).forEach(file=>{
            formData.append("testFiles", file);
        });

        formData.append("model", this.props.data.name);

        this.setState({loadingOverlay: true})

        axios({
            method: "POST",
            url: "http://127.0.0.1:5000/test",
            data: formData,
            headers: {
              "Content-Type": "multipart/form-data"
            }
        }).then(res => {
            console.log(res.data);
            this.setState({loadingOverlay: false}, () => {
                alert("SVM_result: "+res.data.svm+" MLP_result: "+res.data.mlp+" MV5_result: "+res.data.mv5)
                this.props.toggle()
            }
            )
        })
    }

    _handleImageChange(e) {
        e.preventDefault();
        let files = e.target.files;
        console.log(files);
        this.setState({
            files: files
        })
    }
    
    render() {
        return (
            <Modal show={this.props.isOpen} onHide={this.props.toggle}>
                {
                    this.state.loadingOverlay ? <LoadingOverlay /> : <></>
                }
                    <Modal.Header closeButton>
                    <Modal.Title>{ this.props.data.mode != undefined ? this.props.data.mode.toUpperCase() : "" } : { this.props.data.name != undefined ? this.props.data.name.toUpperCase() : "" }</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <form onSubmit={(e)=>this._handleSubmit(e)}>
                            <h5>Test Signature</h5>
                            <input className="" 
                                type="file" 
                                onChange={(e)=>this._handleImageChange(e)} multiple="multiple"/>
                                <hr />
                            <Button className="float-right" variant="outline-primary" type="submit">
                                { this.props.data.mode != undefined ? this.props.data.mode.toUpperCase() : "undefined" }
                            </Button>
                        </form>
                        
                    </Modal.Body>
                </Modal>
            
        )
    }
}

class LoadingOverlay extends Component {
    render() {
        return (
            <div style={{position: 'fixed', top: '0', left: '0', width: '100%', height: '100vh', background: "rgba(0, 0, 0, 0.8)", zIndex: '999', display: 'grid', justifyContent: 'center', alignContent: 'center', gridGap: '1em'}}>
              <Spinner animation="border" variant="primary" style={{ width: '5rem', height: '5rem', justifySelf: 'center'}} />
              <text style={{ justifySelf: 'center', textShadow: "rgba(0, 0, 0, 0.8)"}}>L o a d i n g . . .</text>
            </div>
        )
    }
}

export default Dashboard
