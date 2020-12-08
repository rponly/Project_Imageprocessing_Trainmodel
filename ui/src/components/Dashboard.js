import React, { Component } from 'react'
import { Container, Row, Col} from 'react-bootstrap'
export class Dashboard extends Component {
    render() {
        return (
            <Container fluid className="text-center m-1">
                <Row>
                    <Col>
                        <h5>Select Model</h5>
                    </Col>
                </Row>
            </Container>
        )
    }
}

export default Dashboard
