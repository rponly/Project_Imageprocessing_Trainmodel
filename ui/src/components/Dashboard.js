import React, { Component } from 'react'
import { Container, Row, Col, Card, Button, Modal } from 'react-bootstrap'
import '../assets/main.css'

export class Dashboard extends Component {
    constructor(props) {
        super(props)
        this.state = {
            models: [
                {
                    name: "model one"
                },
                {
                    name: "model two"
                },
                {
                    name: "model three"
                }
            ],
            isModelModalOpen: false,
            dataToPassToModal: {
                mode: null,
                name: null
            },
            loading: true
        }
    }

    componentDidMount() {
        this.setState({loading: false})
    }

    _handleAddNewModel = () => {
        alert("add new model")
    }

    toggleModelModal = (data) => {
        if(data)
            this.setState({isModelModalOpen: !this.state.isModelModalOpen, dataToPassToModal: data})
        else
        this.setState({isModelModalOpen: !this.state.isModelModalOpen})
    }

    render() {
        if(this.state.loading)
            return (<></>)
        return (
            <Container fluid>
                <ModelModal isOpen={this.state.isModelModalOpen} toggle={this.toggleModelModal} data={this.state.dataToPassToModal}/>
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
                                            <Button className="mx-1" variant="outline-success" onClick={() => this.toggleModelModal({mode: 'test', name: model.name})}>Test</Button>
                                        </Card.Footer>
                                    </Card>
                                )
                            })
                        }
                        { /** Add new model */}
                            <Card className="m-1 text-center add-new" style={{ width: "250px" }} onClick={this._handleAddNewModel}>
                                <Card.Body>
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
            file: {},
            imagePreviewUrl: ""
        }
    }

    _handleSubmit(e) {
        e.preventDefault();
        // TODO: do something with -> this.state.file
        console.log('handle uploading-', this.state.file);
    }

    _handleImageChange(e) {
        e.preventDefault();
    
        let reader = new FileReader();
        let file = e.target.files[0];
    
        reader.onloadend = () => {
          this.setState({
            file: file,
            imagePreviewUrl: reader.result
          });
        }
    
        reader.readAsDataURL(file)
    }

    _handleClose = () => {
        this.setState({
            file: {},
            imagePreviewUrl: ""
        }, () => this.props.toggle())
    }

    render() {
        return (
            <Modal show={this.props.isOpen} onHide={this._handleClose}>
                <Modal.Header closeButton>
                <Modal.Title>{ this.props.data.mode != undefined ? this.props.data.mode.toUpperCase() : "" } : { this.props.data.name != undefined ? this.props.data.name.toUpperCase() : "" }</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <div className="d-flex justify-content-center">
                        {
                            this.state.imagePreviewUrl != "" ? <img style={{maxWidth: "100%", maxHeight: "100%"}} src={this.state.imagePreviewUrl} /> : <h6>Please select an image</h6>
                        }
                    </div>
                </Modal.Body>
                <Modal.Footer className="d-flex justify-content-center">
                    <form onSubmit={(e)=>this._handleSubmit(e)}>
                        <input className="" 
                            type="file" 
                            onChange={(e)=>this._handleImageChange(e)} />
                        <Button variant="outline-primary" onClick={this._handleClose}>
                            { this.props.data.mode != undefined ? this.props.data.mode.toUpperCase() : "undefined" }
                        </Button>
                    </form>
                </Modal.Footer>
            </Modal>
        )
    }
}

export default Dashboard
